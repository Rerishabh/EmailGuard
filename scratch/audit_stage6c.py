import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, average_precision_score

print("=" * 80)
print("STAGE 6C: INDEPENDENT ARTIFACT, METRIC & INTEGRITY AUDIT")
print("=" * 80)

# ============================================================
# 1. PARTITION INTEGRITY AND CHECKSUMS
# ============================================================

train_path = "dataset/train_split_indices.npy"
test_path = "dataset/test_split_indices.npy"

train_indices = np.load(train_path)
test_indices = np.load(test_path)

assert len(train_indices) == 2560, f"Expected 2560 training indices, got {len(train_indices)}"
assert len(test_indices) == 641, f"Expected 641 test indices, got {len(test_indices)}"
assert len(np.unique(train_indices)) == 2560, "Duplicate training indices!"
assert len(np.unique(test_indices)) == 641, "Duplicate test indices!"
assert set(train_indices).isdisjoint(set(test_indices)), "CRITICAL: Training and test partitions overlap!"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

train_hash = sha256_file(train_path)
test_hash = sha256_file(test_path)

EXPECTED_TRAIN_HASH = "84a9ef640ccc54711068d52c1f22d767aa800142c0f7b4822014855158e61d91"
EXPECTED_TEST_HASH = "bb1d408a89a7f53177defd1e4cd11d1036cccb577e3bc670a13c0d170210a0f5"

assert train_hash == EXPECTED_TRAIN_HASH, "Training index SHA256 mismatch!"
assert test_hash == EXPECTED_TEST_HASH, "Test index SHA256 mismatch!"

print("\n[PASS] Partition integrity")
print(f"  Training records : {len(train_indices)}")
print(f"  Test records     : {len(test_indices)}")
print(f"  Train SHA256     : {train_hash}")
print(f"  Test SHA256      : {test_hash}")
print("  Train/test overlap: ZERO")

# ============================================================
# 2. LOAD STAGE 6C ARTIFACTS
# ============================================================

folds_df = pd.read_csv("results/stage6c_tuning_folds.csv")
nested_df = pd.read_csv("results/stage6c_nested_oof_scores.csv")
global_cv = pd.read_csv("results/stage6c_global_cv_summary.csv")
calib_df = pd.read_csv("results/stage6c_calibration_oof.csv")
sweep_df = pd.read_csv("results/stage6c_threshold_sweep.csv")

with open("results/stage6c_best_candidates.json", "r", encoding="utf-8") as f:
    best_cand = json.load(f)

# ============================================================
# 3. CHECKPOINT WORKLOAD AND SCOPE-AWARE KEY VALIDATION
# ============================================================

assert len(folds_df) == 2130, f"Expected 2130 tuning records, got {len(folds_df)}"

essential_columns = ["evaluation_scope", "candidate_name", "config_id"]
assert not folds_df[essential_columns].isna().any().any(), "Null values in essential identifiers!"

expected_scopes = {
    "nested_inner": 1680,
    "nested_outer": 15,
    "global_cv": 420,
    "calibration_oof": 15
}
actual_scopes = folds_df["evaluation_scope"].value_counts().to_dict()
assert actual_scopes == expected_scopes, f"Unexpected scope distribution: {actual_scopes}"

scope_rules = {
    "nested_inner": {
        "required": ["outer_fold", "inner_fold"],
        "nullable": ["fold"]
    },
    "nested_outer": {
        "required": ["outer_fold"],
        "nullable": ["inner_fold", "fold"]
    },
    "global_cv": {
        "required": ["fold"],
        "nullable": ["outer_fold", "inner_fold"]
    },
    "calibration_oof": {
        "required": ["fold"],
        "nullable": ["outer_fold", "inner_fold"]
    }
}

for scope, rules in scope_rules.items():
    grp = folds_df[folds_df["evaluation_scope"] == scope]
    for col in rules["required"]:
        assert grp[col].notna().all(), f"{scope}: Unexpected null in {col}"
    for col in rules["nullable"]:
        assert grp[col].isna().all(), f"{scope}: Unexpected non-null in {col}"

key_columns = ["evaluation_scope", "candidate_name", "outer_fold", "inner_fold", "fold", "config_id"]
assert not folds_df.duplicated(subset=key_columns).any(), "Duplicate evaluation keys found!"

print("\n[PASS] Scope-aware evaluation key validation")
for scope, expected in expected_scopes.items():
    print(f"  {scope}: {expected}/{expected} records")
print("[PASS] All 2,130 evaluation records accounted for with zero duplicates.")

# ============================================================
# 4. PIPELINE C & B GLOBAL CV CONFIGURATION RANKING
# ============================================================

c_configs = global_cv[global_cv["candidate_name"] == "pipeline_c_linear_svc"].sort_values("f1_spam_mean", ascending=False)
b_configs = global_cv[global_cv["candidate_name"] == "pipeline_b_linear_svc"].sort_values("f1_spam_mean", ascending=False)

assert len(c_configs) == 48, f"Expected 48 Pipeline C configurations, got {len(c_configs)}"
assert len(b_configs) == 16, f"Expected 16 Pipeline B configurations, got {len(b_configs)}"

print("\n" + "=" * 80)
print("PIPELINE C: TOP 5 GLOBAL CV CONFIGURATIONS")
print("=" * 80)
for rank, (_, row) in enumerate(c_configs.head(5).iterrows(), 1):
    print(f"Rank {rank} | Config: {row['config_id']} | F1: {row['f1_spam_mean']:.6f} | FPR: {row['fpr_mean']:.6f}")
    print(f"  Params: {row['params_json']}")

print("\n" + "=" * 80)
print("PIPELINE B: TOP 3 GLOBAL CV CONFIGURATIONS")
print("=" * 80)
for rank, (_, row) in enumerate(b_configs.head(3).iterrows(), 1):
    print(f"Rank {rank} | Config: {row['config_id']} | F1: {row['f1_spam_mean']:.6f} | FPR: {row['fpr_mean']:.6f}")
    print(f"  Params: {row['params_json']}")

# ============================================================
# 5. VERIFY NESTED OOF AVERAGE PRECISION
# ============================================================

assert len(nested_df) == 7680, f"Expected 7680 nested OOF rows, got {len(nested_df)}"

print("\n" + "=" * 80)
print("NESTED OOF AVERAGE PRECISION VERIFICATION")
print("=" * 80)

for candidate, grp in nested_df.groupby("candidate_name"):
    assert len(grp) == 2560, f"{candidate}: Expected 2560 nested OOF rows"
    assert grp["record_index"].is_unique, f"{candidate}: Duplicate nested OOF indices"
    
    indices = grp["record_index"].to_numpy()
    assert set(indices) == set(range(2560)), f"{candidate}: Nested slice indices mismatch"
    
    reconstructed = train_indices[indices]
    assert set(reconstructed) == set(train_indices), f"{candidate}: Nested OOF does not cover training partition"
    assert set(reconstructed).isdisjoint(set(test_indices)), f"{candidate}: Test leakage detected"

    scores = grp["continuous_score"].to_numpy()
    labels = grp["y_true"].to_numpy()

    assert np.isfinite(scores).all(), f"{candidate}: Non-finite nested OOF scores"
    assert set(np.unique(labels)) == {0, 1}, f"{candidate}: Missing class in nested OOF labels"

    calculated_ap = average_precision_score(labels, scores)
    stored_ap = best_cand["candidates"][candidate]["nested_oof_default_metrics"]["pr_auc_ap"]

    assert abs(calculated_ap - stored_ap) < 1e-4, f"{candidate}: Nested AP mismatch: calc={calculated_ap}, json={stored_ap}"
    print(f"[PASS] {candidate}: Calculated AP={calculated_ap:.6f}, JSON AP={stored_ap:.6f}")

# ============================================================
# 6. CALIBRATION OOF INDEX, METRIC AND THRESHOLD AUDIT
# ============================================================

assert len(calib_df) == 7680, f"Expected 7680 calibration rows, got {len(calib_df)}"

print("\n" + "=" * 80)
print("CALIBRATION OOF VERIFICATION")
print("=" * 80)

for candidate, grp in calib_df.groupby("candidate_name"):
    assert len(grp) == 2560, f"{candidate}: Expected 2560 calibration records"
    assert grp["record_index"].is_unique, f"{candidate}: Duplicate record_index values"

    rec_indices = grp["record_index"].to_numpy()
    assert np.issubdtype(rec_indices.dtype, np.integer), f"{candidate}: record_index must be integer"
    assert set(rec_indices) == set(range(2560)), f"{candidate}: Expected slice positions 0..2559"
    assert np.all((rec_indices >= 0) & (rec_indices < len(train_indices))), f"{candidate}: Slice out of bounds"

    reconstructed = train_indices[rec_indices]
    assert set(reconstructed) == set(train_indices), f"{candidate}: Incomplete training coverage"
    assert set(reconstructed).isdisjoint(set(test_indices)), f"{candidate}: CRITICAL TEST DATA LEAKAGE"

    labels = grp["y_true"].to_numpy()
    scores = grp["continuous_score"].to_numpy()

    assert np.isfinite(scores).all(), f"{candidate}: Non-finite calibration scores"
    assert set(np.unique(labels)) == {0, 1}, f"{candidate}: Missing class in calibration labels"

    if "knn" in candidate:
        assert np.all((scores >= 0.0) & (scores <= 1.0)), f"{candidate}: KNN probabilities outside [0, 1]"

    candidate_json = best_cand["candidates"][candidate]
    metrics = candidate_json["calibration_oof_metrics"]
    threshold = metrics["selected_threshold"]

    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    cm = metrics["confusion_matrix"]
    assert tn == cm["tn"], f"{candidate}: TN mismatch"
    assert fp == cm["fp"], f"{candidate}: FP mismatch"
    assert fn == cm["fn"], f"{candidate}: FN mismatch"
    assert tp == cm["tp"], f"{candidate}: TP mismatch"

    assert abs(precision - metrics["calibrated_precision"]) < 1e-4, f"{candidate}: Precision mismatch"
    assert abs(recall - metrics["calibrated_recall"]) < 1e-4, f"{candidate}: Recall mismatch"
    assert abs(f1 - metrics["calibrated_f1"]) < 1e-4, f"{candidate}: F1 mismatch"
    assert abs(fpr - metrics["calibrated_fpr"]) < 1e-4, f"{candidate}: FPR mismatch"
    assert fpr <= 0.005001, f"{candidate}: FPR limit violated: {fpr}"

    # Verify selected threshold in threshold sweep
    candidate_sweep = sweep_df[sweep_df["candidate_name"] == candidate].copy()
    selected_mask = candidate_sweep["selected_threshold"].astype(str).str.strip().str.lower().isin(["true", "1"])
    selected_rows = candidate_sweep[selected_mask]
    assert len(selected_rows) == 1, f"{candidate}: Expected 1 selected threshold row"
    sweep_threshold = float(selected_rows.iloc[0]["threshold"])
    assert abs(sweep_threshold - threshold) < 1e-6, f"{candidate}: Sweep threshold mismatch"

    print(f"\nCandidate: {candidate}")
    print(f"  Params            : {candidate_json['global_best_parameters']}")
    print(f"  Default threshold : {metrics['default_threshold']}")
    print(f"  Calibrated tau*   : {threshold:.8f}")
    print(f"  Confusion matrix  : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Precision         : {precision:.6f}")
    print(f"  Recall            : {recall:.6f}")
    print(f"  F1                : {f1:.6f}")
    print(f"  FPR               : {fpr:.6f}")
    print("  [PASS] Verified against stored JSON and sweep CSV.")

# ============================================================
# 7. FINAL VERDICT
# ============================================================

print("\n" + "=" * 80)
print("STAGE 6C INDEPENDENT AUDIT COMPLETED SUCCESSFULLY")
print("=" * 80)
print("All 2,130 evaluations, 7,680 nested OOF scores, and 7,680 calibration scores match.")
print("Untouched test set: strictly quarantined (N=641, zero leakage).")
