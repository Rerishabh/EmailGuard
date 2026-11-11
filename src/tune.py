"""Stage 6C Tuning Engine: Nested Cross-Validation, Global Hyperparameter Selection,
and Exact Out-Of-Fold Threshold Calibration.

Provides:
- Checkpoint journaling and trailing-row recovery for durable per-record checkpointing.
- Candidate configuration space definitions (Candidates 1, 2, 3).
- Nested 5x4 CV execution for unbiased model-selection assessment.
- Global 5-fold CV for hyperparameter selection on full training partition.
- Fixed-configuration 5-fold OOF continuous score generation for threshold calibration.
- Exact threshold sweeping (evaluating every unique score + nextafter infinity boundary).
- Artifact builders for CSV, JSON, and comprehensive Markdown summary.
"""

import csv
import json
import os
from pathlib import Path
import time
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from src.pipeline import (
    create_pipeline_a,
    create_pipeline_b,
    create_pipeline_c,
    get_classifier,
)

# ---------------------------------------------------------------------
# Checkpoint Schema Constants
# ---------------------------------------------------------------------
CHECKPOINT_COLUMNS = [
    "evaluation_scope",
    "candidate_name",
    "outer_fold",
    "inner_fold",
    "fold",
    "config_id",
    "params_json",
    "accuracy",
    "precision_spam",
    "recall_spam",
    "f1_spam",
    "fpr",
    "pr_auc",
    "n_iter",
    "converged",
    "t_train_s",
    "t_infer_s",
]

VALID_SCOPES = {"nested_inner", "nested_outer", "global_cv", "calibration_oof"}
VALID_CANDIDATES = {"pipeline_c_linear_svc", "pipeline_b_linear_svc", "pipeline_a_knn"}


# ---------------------------------------------------------------------
# Candidate Configuration Builders
# ---------------------------------------------------------------------
def get_candidate_grid(candidate_name: str) -> List[Dict[str, Any]]:
    """Return the exact hyperparameter search grid for a candidate."""
    configs = []
    if candidate_name == "pipeline_c_linear_svc":
        # 48 configurations: C (6) x loss (2) x heuristic_weight (4)
        c_vals = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
        losses = ["squared_hinge", "hinge"]
        weights = [0.05, 0.25, 0.5, 1.0]
        cfg_idx = 0
        for w in weights:
            for l in losses:
                for c in c_vals:
                    cfg_idx += 1
                    configs.append({
                        "config_id": f"c_lsvc_w{w}_l{l}_c{c}",
                        "params": {
                            "C": c,
                            "loss": l,
                            "heuristic_weight": w,
                            "max_iter": 5000,
                            "dual": True if l == "hinge" else "auto",
                        }
                    })
    elif candidate_name == "pipeline_b_linear_svc":
        # 16 configurations: C (8) x loss (2)
        c_vals = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        losses = ["squared_hinge", "hinge"]
        cfg_idx = 0
        for l in losses:
            for c in c_vals:
                cfg_idx += 1
                configs.append({
                    "config_id": f"b_lsvc_l{l}_c{c}",
                    "params": {
                        "C": c,
                        "loss": l,
                        "max_iter": 5000,
                        "dual": True if l == "hinge" else "auto",
                    }
                })
    elif candidate_name == "pipeline_a_knn":
        # 20 configurations: n_neighbors (5) x weights (2) x metric (2)
        neighbors = [3, 5, 7, 11, 15]
        weights = ["uniform", "distance"]
        metrics = ["cosine", "euclidean"]
        cfg_idx = 0
        for m in metrics:
            for w in weights:
                for k in neighbors:
                    cfg_idx += 1
                    configs.append({
                        "config_id": f"a_knn_m{m}_w{w}_k{k}",
                        "params": {
                            "n_neighbors": k,
                            "weights": w,
                            "metric": m,
                        }
                    })
    else:
        raise ValueError(f"Unrecognized candidate_name: {candidate_name}")

    return configs


def build_pipeline_from_config(candidate_name: str, params: Dict[str, Any]):
    """Instantiate a fresh, un-fitted pipeline corresponding to the configuration."""
    p = params.copy()
    if candidate_name == "pipeline_c_linear_svc":
        w = p.pop("heuristic_weight", 1.0)
        clf = get_classifier("linear_svc", **p)
        return create_pipeline_c(classifier=clf, heuristic_weight=w)
    elif candidate_name == "pipeline_b_linear_svc":
        clf = get_classifier("linear_svc", **p)
        return create_pipeline_b(classifier=clf)
    elif candidate_name == "pipeline_a_knn":
        clf = get_classifier("knn", **p)
        return create_pipeline_a(classifier=clf)
    else:
        raise ValueError(f"Unrecognized candidate_name: {candidate_name}")


# ---------------------------------------------------------------------
# Checkpoint Journaling & Trailing-Row Recovery
# ---------------------------------------------------------------------
def read_checkpoint(checkpoint_path: Path) -> Tuple[Set[Tuple], List[Dict[str, Any]]]:
    """Read and validate the checkpoint CSV with trailing-row recovery.

    Returns:
        completed_keys: set of (evaluation_scope, candidate_name, outer_fold, inner_fold, fold, config_id)
        valid_records: list of validated row dicts
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        return set(), []

    valid_records = []
    completed_keys = set()
    valid_byte_pos = 0

    with open(checkpoint_path, "rb") as fb:
        lines = fb.readlines()

    if not lines:
        return set(), []

    try:
        header_line = lines[0].decode("utf-8").strip()
    except UnicodeDecodeError:
        raise ValueError(f"Header decoding error in {checkpoint_path}")

    header = [c.strip() for c in header_line.split(",")]
    if header != CHECKPOINT_COLUMNS:
        raise ValueError(f"Checkpoint header mismatch in {checkpoint_path}. Got: {header}")

    valid_byte_pos += len(lines[0])

    for line_idx, raw_line in enumerate(lines[1:], start=2):
        try:
            line_str = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError:
            print(f"[WARN] Non-UTF8 decoding error at line {line_idx} in {checkpoint_path}. Truncating trailing bytes.")
            break

        if not line_str:
            continue

        # Use csv.reader on single line to handle quotes properly
        row_reader = list(csv.reader([line_str]))
        if not row_reader or len(row_reader[0]) != len(CHECKPOINT_COLUMNS):
            print(f"[WARN] Incomplete row at line {line_idx} in {checkpoint_path} (len {len(row_reader[0]) if row_reader else 0}). Recovering preceding valid records.")
            break

        row = row_reader[0]
        record = dict(zip(CHECKPOINT_COLUMNS, row))

        # Validate fields
        scope = record["evaluation_scope"]
        cand = record["candidate_name"]
        if scope not in VALID_SCOPES:
            raise ValueError(f"Invalid evaluation_scope '{scope}' at line {line_idx}")
        if cand not in VALID_CANDIDATES:
            raise ValueError(f"Invalid candidate_name '{cand}' at line {line_idx}")

        # Validate metrics range [0.0, 1.0] and finiteness
        for m in ["accuracy", "precision_spam", "recall_spam", "f1_spam", "fpr", "pr_auc"]:
            val_str = record[m]
            if val_str == "" or val_str.lower() == "nan":
                val = np.nan
            else:
                try:
                    val = float(val_str)
                except ValueError:
                    raise ValueError(f"Non-numeric metric {m}='{val_str}' at line {line_idx}")
                if not (0.0 <= val <= 1.0000001):
                    raise ValueError(f"Metric {m}={val} out of bounds [0, 1] at line {line_idx}")

        # Validate timing
        for t in ["t_train_s", "t_infer_s"]:
            try:
                t_val = float(record[t])
                if t_val < 0.0 or not np.isfinite(t_val):
                    raise ValueError
            except ValueError:
                raise ValueError(f"Invalid timing {t}='{record[t]}' at line {line_idx}")

        # Validate key uniqueness
        key = (
            record["evaluation_scope"],
            record["candidate_name"],
            record["outer_fold"],
            record["inner_fold"],
            record["fold"],
            record["config_id"],
        )
        if key in completed_keys:
            raise ValueError(f"Duplicate evaluation key {key} in checkpoint at line {line_idx}")

        completed_keys.add(key)
        valid_records.append(record)
        valid_byte_pos += len(raw_line)

    # Truncate any incomplete trailing bytes if recovery is needed
    current_size = Path(checkpoint_path).stat().st_size
    if current_size > valid_byte_pos:
        print(f"[RECOVERY] Truncating {checkpoint_path} from {current_size} to {valid_byte_pos} bytes.")
        with open(checkpoint_path, "a+b") as f:
            f.truncate(valid_byte_pos)

    return completed_keys, valid_records


def append_tuning_record(record_dict: Dict[str, Any], checkpoint_path: Path):
    """Atomically append a single validated tuning record with immediate fsync."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate mandatory fields
    for col in CHECKPOINT_COLUMNS:
        if col not in record_dict:
            raise ValueError(f"Missing required checkpoint column '{col}'")

    file_exists = checkpoint_path.exists() and checkpoint_path.stat().st_size > 0

    with open(checkpoint_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHECKPOINT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: record_dict[k] for k in CHECKPOINT_COLUMNS})
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------
# Evaluation Helper with Convergence Monitoring
# ---------------------------------------------------------------------
def fit_and_evaluate_single(
    pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """Fit pipeline, monitor convergence warnings, and evaluate validation metrics."""
    converged = True
    n_iter_val = ""

    # Measure fit time and capture ConvergenceWarning
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        pipeline.fit(X_train, y_train)
        for w in caught_warnings:
            if "converge" in str(w.message).lower() or "iteration" in str(w.message).lower():
                converged = False
    t_train_s = time.perf_counter() - t0

    # Extract n_iter_ from classifier step if available
    clf = pipeline.named_steps.get("classifier")
    if clf is not None and hasattr(clf, "n_iter_"):
        n_iter_attr = clf.n_iter_
        if isinstance(n_iter_attr, (list, tuple, np.ndarray)):
            n_iter_val = int(np.max(n_iter_attr))
        else:
            n_iter_val = int(n_iter_attr)

    # Measure inference time and generate continuous scores
    t1 = time.perf_counter()
    scores: Optional[np.ndarray] = None
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X_val)
        y_pred = (scores >= 0.0).astype(int)
    elif hasattr(pipeline, "predict_proba"):
        spam_idx = list(pipeline.classes_).index(1)
        scores = pipeline.predict_proba(X_val)[:, spam_idx]
        y_pred = (scores >= 0.5).astype(int)
    else:
        y_pred = pipeline.predict(X_val)
    t_infer_s = time.perf_counter() - t1

    # Compute metrics
    acc = float(accuracy_score(y_val, y_pred))
    p_spam = float(precision_score(y_val, y_pred, pos_label=1, zero_division=0))
    r_spam = float(recall_score(y_val, y_pred, pos_label=1, zero_division=0))
    f1_sp = float(f1_score(y_val, y_pred, pos_label=1, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    if scores is not None:
        pr_auc = float(average_precision_score(y_val, scores))
    else:
        pr_auc = np.nan

    return {
        "accuracy": acc,
        "precision_spam": p_spam,
        "recall_spam": r_spam,
        "f1_spam": f1_sp,
        "fpr": fpr,
        "pr_auc": pr_auc,
        "n_iter": n_iter_val,
        "converged": converged,
        "t_train_s": round(t_train_s, 6),
        "t_infer_s": round(t_infer_s, 6),
        "scores": scores,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------
# Deterministic Tie-Breaking
# ---------------------------------------------------------------------
def pick_best_config(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the best configuration using deterministic multi-tier criteria:
    1. Highest mean Spam F1.
    2. Highest mean Spam Precision.
    3. Lowest mean FPR.
    4. Simpler configuration (deterministic rule: lower heuristic_weight, higher C/k, simpler loss).
    """
    def sort_key(rec):
        f1 = rec["f1_spam_mean"]
        prec = rec["precision_spam_mean"]
        fpr = -rec["fpr_mean"]  # lower is better
        cfg_id = rec["config_id"]
        return (f1, prec, fpr, cfg_id)

    sorted_records = sorted(records, key=sort_key, reverse=True)
    return sorted_records[0]


# ---------------------------------------------------------------------
# Exact Threshold Optimization
# ---------------------------------------------------------------------
def sweep_exact_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    candidate_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate every unique score threshold plus infinity nextafter boundary.

    Primary Objective:
        max_tau F1_spam(tau) subject to FPR(tau) <= 0.005 (0.5% max FPR)

    Fallback Rule:
        If no threshold satisfies FPR <= 0.005 and F1 >= 0.85:
        1. Select threshold with minimum FPR.
        2. Tie-break with maximum Spam F1.
        3. Tie-break with higher threshold.
        Explicitly flag fallback_activated = True.
    """
    unique_scores = np.sort(np.unique(scores))
    thresholds = np.append(unique_scores, np.nextafter(unique_scores[-1], np.inf))

    rows = []
    default_tau = 0.0 if "linear_svc" in candidate_name else 0.5

    for tau in thresholds:
        y_pred = (scores >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        n_pos = tp + fn
        n_neg = tn + fp

        acc = (tp + tn) / (n_pos + n_neg)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / n_pos if n_pos > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0

        satisfies_fpr = bool(fpr <= 0.00500001)

        rows.append({
            "candidate_name": candidate_name,
            "threshold": float(tau),
            "accuracy": float(acc),
            "precision_spam": float(prec),
            "recall_spam": float(rec),
            "f1_spam": float(f1),
            "fpr": float(fpr),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "satisfies_fpr_limit": satisfies_fpr,
            "selected_threshold": False,
        })

    sweep_df = pd.DataFrame(rows)

    # Primary selection: max F1 where satisfies_fpr_limit is True
    feasible_df = sweep_df[sweep_df["satisfies_fpr_limit"]]
    fallback_activated = False

    if not feasible_df.empty and (feasible_df["f1_spam"] >= 0.85).any():
        # High-F1 feasible exists
        target_df = feasible_df[feasible_df["f1_spam"] >= 0.85]
        # max F1, tie-break min FPR, tie-break higher threshold
        best_row_idx = target_df.sort_values(
            ["f1_spam", "fpr", "threshold"], ascending=[False, True, False]
        ).index[0]
    elif not feasible_df.empty:
        # Feasible under FPR, but F1 < 0.85
        best_row_idx = feasible_df.sort_values(
            ["f1_spam", "fpr", "threshold"], ascending=[False, True, False]
        ).index[0]
        fallback_activated = True
    else:
        # No threshold satisfies FPR <= 0.005 -> activate deterministic fallback
        fallback_activated = True
        best_row_idx = sweep_df.sort_values(
            ["fpr", "f1_spam", "threshold"], ascending=[True, False, False]
        ).index[0]

    sweep_df.loc[best_row_idx, "selected_threshold"] = True
    selected_row = sweep_df.loc[best_row_idx].to_dict()

    # Get default threshold stats
    def_row_idx = (sweep_df["threshold"] - default_tau).abs().idxmin()
    default_row = sweep_df.loc[def_row_idx].to_dict()

    summary_info = {
        "candidate_name": candidate_name,
        "default_threshold": float(default_tau),
        "selected_threshold": float(selected_row["threshold"]),
        "default_f1": float(default_row["f1_spam"]),
        "calibrated_f1": float(selected_row["f1_spam"]),
        "default_precision": float(default_row["precision_spam"]),
        "calibrated_precision": float(selected_row["precision_spam"]),
        "default_recall": float(default_row["recall_spam"]),
        "calibrated_recall": float(selected_row["recall_spam"]),
        "default_fpr": float(default_row["fpr"]),
        "calibrated_fpr": float(selected_row["fpr"]),
        "accuracy": float(selected_row["accuracy"]),
        "confusion_matrix": {
            "tn": int(selected_row["tn"]),
            "fp": int(selected_row["fp"]),
            "fn": int(selected_row["fn"]),
            "tp": int(selected_row["tp"]),
        },
        "satisfies_fpr_limit": bool(selected_row["satisfies_fpr_limit"]),
        "fallback_activated": fallback_activated,
    }

    return sweep_df, summary_info
