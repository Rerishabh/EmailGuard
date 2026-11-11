"""Model training, hyperparameter tuning, and cross-validation execution module.

Provides primary public functions:
    - evaluate_fold_predictions: Computes per-fold classification metrics.
    - get_completed_folds: Inspects CSV checkpoint and returns validated completed fold keys.
    - append_fold_result: Atomically appends a single completed fold record to disk.
    - run_cross_validation_single: Runs 5-fold CV for one (pipeline, classifier) pair.
    - run_full_cv_matrix: Executes all 18 (3 pipelines × 6 classifiers) CV combinations
                          with resumable per-fold checkpointing, returning fold-level and
                          aggregated summary DataFrames.
"""

import csv
import os
from pathlib import Path
import time
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
from sklearn.model_selection import BaseCrossValidator

from src.pipeline import (
    create_pipeline_a,
    create_pipeline_b,
    create_pipeline_c,
    get_classifier,
)

REQUIRED_METRIC_COLS = [
    "t_train_s",
    "t_infer_s",
    "accuracy",
    "precision_spam",
    "recall_spam",
    "f1_spam",
    "precision_ham",
    "recall_ham",
    "fpr",
    "pr_auc_average_precision",
]

REQUIRED_FOLD_COLS = ["fold", "pipeline", "classifier"] + REQUIRED_METRIC_COLS

VALID_PIPELINES = {"pipeline_a", "pipeline_b", "pipeline_c"}
VALID_CLASSIFIERS = {
    "logistic_regression",
    "multinomial_nb",
    "decision_tree",
    "random_forest",
    "linear_svc",
    "knn",
}
VALID_FOLDS = {0, 1, 2, 3, 4}


def evaluate_fold_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute per-fold classification metrics for binary spam detection.

    Args:
        y_true:  Ground-truth binary labels (0 = Ham, 1 = Spam).
        y_pred:  Hard binary predictions from pipe.predict().
        y_score: Continuous confidence scores for class 1 (Spam); used for PR-AUC.
                 Pass None if the classifier does not support probability estimates
                 or decision functions.

    Returns:
        Dictionary containing:
            accuracy, precision_spam, recall_spam, f1_spam,
            precision_ham, recall_ham, fpr, pr_auc_average_precision
    """
    accuracy = float(accuracy_score(y_true, y_pred))

    precision_spam = float(
        precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    )
    recall_spam = float(
        recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    )
    f1_spam = float(
        f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    )
    precision_ham = float(
        precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    )
    recall_ham = float(
        recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    )

    # False Positive Rate: FP / (FP + TN)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan

    # PR-AUC (Average Precision) for ranking quality under class imbalance
    if y_score is not None:
        pr_auc = float(average_precision_score(y_true, y_score))
    else:
        pr_auc = np.nan

    return {
        "accuracy": accuracy,
        "precision_spam": precision_spam,
        "recall_spam": recall_spam,
        "f1_spam": f1_spam,
        "precision_ham": precision_ham,
        "recall_ham": recall_ham,
        "fpr": fpr,
        "pr_auc_average_precision": pr_auc,
    }


def get_completed_folds(checkpoint_path: Path) -> Set[Tuple[str, str, int]]:
    """Inspect CSV checkpoint file and return validated set of (pipeline, classifier, fold) keys.

    Validates schema, column existence, range checks, and strictly rejects duplicate keys
    or malformed records.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        return set()

    df = pd.read_csv(checkpoint_path)

    # Validate required columns
    missing_cols = [c for c in REQUIRED_FOLD_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Checkpoint file {checkpoint_path} is missing required columns: {missing_cols}"
        )

    completed: Set[Tuple[str, str, int]] = set()
    for idx, row in df.iterrows():
        p = str(row["pipeline"]).strip()
        c = str(row["classifier"]).strip()
        try:
            f = int(row["fold"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid fold index '{row['fold']}' at row {idx} in {checkpoint_path}") from e

        if p not in VALID_PIPELINES:
            raise ValueError(f"Invalid pipeline '{p}' at row {idx} in {checkpoint_path}")
        if c not in VALID_CLASSIFIERS:
            raise ValueError(f"Invalid classifier '{c}' at row {idx} in {checkpoint_path}")
        if f not in VALID_FOLDS:
            raise ValueError(f"Invalid fold number {f} at row {idx} in {checkpoint_path}")

        key = (p, c, f)
        if key in completed:
            raise ValueError(f"Duplicate fold record detected in {checkpoint_path}: {key}")

        # Validate metrics are non-null and numeric
        for col in REQUIRED_METRIC_COLS:
            val = row[col]
            if pd.isna(val) and col != "pr_auc_average_precision":
                raise ValueError(f"Null or NaN value in metric '{col}' at row {idx} for fold {key}")

        completed.add(key)

    return completed


def append_fold_result(fold_dict: Dict[str, Any], checkpoint_path: Path) -> None:
    """Atomically append a single completed fold record to disk.

    Flushes immediately and invokes os.fsync to ensure immediate durability.
    Prevents duplicate entries and verifies read-back integrity.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate all required columns exist in input
    for col in REQUIRED_FOLD_COLS:
        if col not in fold_dict:
            raise ValueError(f"Fold dictionary missing required column '{col}': {fold_dict}")

    key = (str(fold_dict["pipeline"]), str(fold_dict["classifier"]), int(fold_dict["fold"]))

    # Check for duplicate against existing on-disk records
    existing = get_completed_folds(checkpoint_path)
    if key in existing:
        raise ValueError(f"Cannot append duplicate fold record {key} to {checkpoint_path}")

    file_exists = checkpoint_path.exists() and checkpoint_path.stat().st_size > 0

    with open(checkpoint_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_FOLD_COLS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: fold_dict[k] for k in REQUIRED_FOLD_COLS})
        f.flush()
        os.fsync(f.fileno())

    # Read-back verification
    updated = get_completed_folds(checkpoint_path)
    if key not in updated:
        raise IOError(f"Read-back verification failed for fold {key} after writing to {checkpoint_path}")


def run_cross_validation_single(
    pipeline_factory,
    classifier_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: BaseCrossValidator,
    pipeline_name: str = "unknown",
    checkpoint_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run 5-fold stratified cross-validation for one (pipeline, classifier) combination.

    Supports atomic per-fold persistence and resuming from checkpoint.
    """
    splits = list(cv.split(X_train, y_train))
    assert len(splits) == 5, f"Expected 5 splits from CV, got {len(splits)}"

    completed_keys = get_completed_folds(checkpoint_path) if checkpoint_path else set()
    done_folds = {f for (p, c, f) in completed_keys if p == pipeline_name and c == classifier_name}

    fold_results: List[Dict[str, Any]] = []

    for fold_k in range(5):
        if fold_k in done_folds:
            continue

        tr_idx, val_idx = splits[fold_k]
        X_tr = X_train[tr_idx]
        y_tr = y_train[tr_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        clf = clone(get_classifier(classifier_name))
        pipe = pipeline_factory(classifier=clf)

        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        t_train = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = pipe.predict(X_val)

        y_score: Optional[np.ndarray] = None
        if hasattr(pipe, "predict_proba"):
            try:
                spam_idx = list(pipe.classes_).index(1)
                y_score = pipe.predict_proba(X_val)[:, spam_idx]
            except (AttributeError, ValueError):
                y_score = None
        elif hasattr(pipe, "decision_function"):
            try:
                y_score = pipe.decision_function(X_val)
            except AttributeError:
                y_score = None

        t_infer = time.perf_counter() - t1

        metrics = evaluate_fold_predictions(y_val, y_pred, y_score)

        fold_record: Dict[str, Any] = {
            "fold": fold_k,
            "pipeline": pipeline_name,
            "classifier": classifier_name,
            "t_train_s": round(t_train, 6),
            "t_infer_s": round(t_infer, 6),
            **{k: round(v, 8) if not np.isnan(v) else np.nan for k, v in metrics.items()},
        }

        if checkpoint_path:
            append_fold_result(fold_record, checkpoint_path)

        fold_results.append(fold_record)

    return fold_results


def run_full_cv_matrix(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: BaseCrossValidator,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute all 18 (3 pipelines × 6 classifiers) cross-validation combinations.

    Supports atomic per-fold persistence and resumable execution.
    """
    pipeline_registry = {
        "pipeline_a": create_pipeline_a,
        "pipeline_b": create_pipeline_b,
        "pipeline_c": create_pipeline_c,
    }
    classifier_names = [
        "logistic_regression",
        "multinomial_nb",
        "decision_tree",
        "random_forest",
        "linear_svc",
        "knn",
    ]

    splits = list(cv.split(X_train, y_train))
    assert len(splits) == 5, f"Expected 5 splits from CV, got {len(splits)}"

    total_combinations = len(pipeline_registry) * len(classifier_names)
    comb_idx = 0

    for pipe_name, pipe_factory in pipeline_registry.items():
        for clf_name in classifier_names:
            comb_idx += 1
            completed_keys = get_completed_folds(checkpoint_path) if checkpoint_path else set()
            done_folds = {f for (p, c, f) in completed_keys if p == pipe_name and c == clf_name}

            if len(done_folds) == 5:
                print(f"[{comb_idx}/{total_combinations}] SKIP: {pipe_name} + {clf_name} (all 5 folds already completed)", flush=True)
                continue

            missing_folds = [k for k in range(5) if k not in done_folds]
            print(f"[{comb_idx}/{total_combinations}] RUN: {pipe_name} + {clf_name} (executing missing folds: {missing_folds})", flush=True)

            for fold_k in missing_folds:
                tr_idx, val_idx = splits[fold_k]
                X_tr = X_train[tr_idx]
                y_tr = y_train[tr_idx]
                X_val = X_train[val_idx]
                y_val = y_train[val_idx]

                clf = clone(get_classifier(clf_name))
                pipe = pipe_factory(classifier=clf)

                t0 = time.perf_counter()
                pipe.fit(X_tr, y_tr)
                t_train = time.perf_counter() - t0

                t1 = time.perf_counter()
                y_pred = pipe.predict(X_val)

                y_score: Optional[np.ndarray] = None
                if hasattr(pipe, "predict_proba"):
                    try:
                        spam_idx = list(pipe.classes_).index(1)
                        y_score = pipe.predict_proba(X_val)[:, spam_idx]
                    except (AttributeError, ValueError):
                        y_score = None
                elif hasattr(pipe, "decision_function"):
                    try:
                        y_score = pipe.decision_function(X_val)
                    except AttributeError:
                        y_score = None

                t_infer = time.perf_counter() - t1

                metrics = evaluate_fold_predictions(y_val, y_pred, y_score)

                fold_record: Dict[str, Any] = {
                    "fold": fold_k,
                    "pipeline": pipe_name,
                    "classifier": clf_name,
                    "t_train_s": round(t_train, 6),
                    "t_infer_s": round(t_infer, 6),
                    **{k: round(v, 8) if not np.isnan(v) else np.nan for k, v in metrics.items()},
                }

                if checkpoint_path:
                    append_fold_result(fold_record, checkpoint_path)

                print(
                    f"    Fold {fold_k}/4: F1={metrics['f1_spam']:.4f}, "
                    f"Prec={metrics['precision_spam']:.4f}, Rec={metrics['recall_spam']:.4f}, "
                    f"FPR={metrics['fpr']:.4f}, Time={t_train:.2f}s",
                    flush=True,
                )

    # Load all completed folds from checkpoint or build DataFrame
    if checkpoint_path and Path(checkpoint_path).exists():
        folds_df = pd.read_csv(checkpoint_path)
    else:
        raise RuntimeError("No checkpoint file available to generate summary.")

    assert len(folds_df) == 90, f"Expected exactly 90 fold records, found {len(folds_df)}"

    # Sort deterministically
    folds_df["fold"] = folds_df["fold"].astype(int)
    folds_df = folds_df.sort_values(["pipeline", "classifier", "fold"]).reset_index(drop=True)

    # Compute summary DataFrame
    summary_records = []
    for (pipe_name, clf_name), grp in folds_df.groupby(["pipeline", "classifier"], sort=False):
        record: Dict[str, Any] = {
            "pipeline": pipe_name,
            "classifier": clf_name,
        }
        for col in REQUIRED_METRIC_COLS:
            vals = grp[col].dropna()
            record[f"{col}_mean"] = round(float(vals.mean()), 6) if len(vals) > 0 else np.nan
            record[f"{col}_std"] = round(float(vals.std(ddof=1)), 6) if len(vals) > 1 else np.nan
        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)
    assert len(summary_df) == 18, f"Expected 18 summary rows, got {len(summary_df)}"

    return folds_df, summary_df
