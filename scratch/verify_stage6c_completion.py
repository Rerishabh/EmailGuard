import os
import sys
import json
import hashlib
import csv
import pandas as pd
import numpy as np

print("=== 1. VERIFY ALL 7 ARTIFACTS EXIST & GET STATS ===")
artifacts = [
    "results/stage6c_tuning_folds.csv",
    "results/stage6c_nested_oof_scores.csv",
    "results/stage6c_global_cv_summary.csv",
    "results/stage6c_calibration_oof.csv",
    "results/stage6c_threshold_sweep.csv",
    "results/stage6c_best_candidates.json",
    "results/stage6c_tuning_summary.md",
]

for art in artifacts:
    p = os.path.join(".", art)
    exists = os.path.exists(p)
    size = os.path.getsize(p) if exists else 0
    line_count = 0
    if exists:
        with open(p, "rb") as f:
            line_count = sum(1 for _ in f)
    print(f"{art}: exists={exists}, size_bytes={size}, total_lines={line_count}")

print("\n=== 2. CHECKPOINT STRUCTURAL AUDIT ===")
with open("results/stage6c_tuning_folds.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    chk_rows = list(reader)

print(f"Total checkpoint data rows: {len(chk_rows)}")
keys = set()
dup_count = 0
null_counts = {k: 0 for k in chk_rows[0].keys()}
for r in chk_rows:
    k = (r["evaluation_scope"], r["candidate_name"], r["outer_fold"], r["inner_fold"], r["fold"], r["config_id"])
    if k in keys:
        dup_count += 1
    keys.add(k)
    for col, val in r.items():
        if val is None or val == "":
            if col not in ["outer_fold", "inner_fold", "fold"]:
                null_counts[col] += 1

print(f"Unique evaluation keys: {len(keys)}")
print(f"Duplicate keys: {dup_count}")
print(f"Unexpected nulls: {[k for k, v in null_counts.items() if v > 0]}")

df_chk = pd.DataFrame(chk_rows)
print("\nEvaluation-scope coverage:")
print(df_chk["evaluation_scope"].value_counts().to_string())

print("\nCandidate-wise coverage:")
print(df_chk.groupby(["evaluation_scope", "candidate_name"]).size().to_string())

print("\n=== 3. NESTED OOF PREDICTIONS AUDIT ===")
df_nested = pd.read_csv("results/stage6c_nested_oof_scores.csv")
print(f"Total nested OOF rows: {len(df_nested)}")
print(f"Columns: {list(df_nested.columns)}")
print("Candidate counts:")
print(df_nested["candidate_name"].value_counts().to_string())
print(f"Missing values: {df_nested.isnull().sum().to_dict()}")

train_indices = np.load("dataset/train_split_indices.npy")
train_idx_set = set(train_indices)
print(f"Train split indices count: {len(train_indices)}")

for cname, grp in df_nested.groupby("candidate_name"):
    c_set = set(grp["record_index"])
    matches = (c_set == train_idx_set)
    print(f"  {cname}: count={len(grp)}, unique_idx={len(c_set)}, matches_train_split={matches}")
    print(f"    score_min={grp['continuous_score'].min():.4f}, score_max={grp['continuous_score'].max():.4f}, mean={grp['continuous_score'].mean():.4f}, std={grp['continuous_score'].std():.4f}")

print("\nNested OOF First 5 rows:")
print(df_nested.head(5).to_string())
print("\nNested OOF Last 5 rows:")
print(df_nested.tail(5).to_string())

print("\n=== 4. CALIBRATION OOF AUDIT ===")
df_calib = pd.read_csv("results/stage6c_calibration_oof.csv")
print(f"Total calib OOF rows: {len(df_calib)}")
print(f"Columns: {list(df_calib.columns)}")
print("Candidate counts:")
print(df_calib["candidate_name"].value_counts().to_string())

for cname, grp in df_calib.groupby("candidate_name"):
    c_set = set(grp["record_index"])
    matches = (c_set == train_idx_set)
    print(f"  {cname}: count={len(grp)}, unique_idx={len(c_set)}, matches_train_split={matches}")
    print(f"    score_min={grp['continuous_score'].min():.4f}, score_max={grp['continuous_score'].max():.4f}, mean={grp['continuous_score'].mean():.4f}, std={grp['continuous_score'].std():.4f}")

with open("results/stage6c_best_candidates.json", "r", encoding="utf-8") as f:
    best_cand_json = json.load(f)

print("\nSelected Thresholds from best_candidates.json:")
for cname, cdata in best_cand_json["candidates"].items():
    cal = cdata["calibration_oof_metrics"]
    print(f"  {cname}: def_tau={cal['default_threshold']}, calib_tau={cal['selected_threshold']:.6f}, calib_f1={cal['calibrated_f1']:.4f}, calib_fpr={cal['calibrated_fpr']:.4f}, satisfies_fpr={cal['satisfies_fpr_limit']}")

print("\n=== 5. GLOBAL CV AUDIT ===")
df_global = pd.read_csv("results/stage6c_global_cv_summary.csv")
print(f"Total global summary rows: {len(df_global)}")
for cname, grp in df_global.groupby("candidate_name"):
    best_row = grp.sort_values("f1_spam_mean", ascending=False).iloc[0]
    print(f"  {cname}: best_cfg={best_row['config_id']}, f1_mean={best_row['f1_spam_mean']:.4f}, prec_mean={best_row['precision_spam_mean']:.4f}, fpr_mean={best_row['fpr_mean']:.4f}")

chk_global = df_chk[df_chk["evaluation_scope"] == "global_cv"]
non_conv = chk_global[chk_global["converged"].astype(str).str.lower().isin(["false", "0"])]
print(f"Total global CV evaluations: {len(chk_global)}")
print(f"Total non-converged evaluations (reached max_iter): {len(non_conv)}")
if len(non_conv) > 0:
    print("Non-converged breakdown by candidate and config:")
    print(non_conv.groupby(["candidate_name", "config_id"]).size().to_string())

print("\n=== 6. PRELIMINARY INTEGRITY VERIFICATION ===")
def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()
def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

print("spam_assassin.csv MD5:", md5("dataset/spam_assassin.csv"))
print("spam_assassin_cleaned.csv MD5:", md5("dataset/spam_assassin_cleaned.csv"))
print("train_split_indices.npy SHA256:", sha256("dataset/train_split_indices.npy"))
print("test_split_indices.npy SHA256:", sha256("dataset/test_split_indices.npy"))
print("src/preprocessing.py lines:", len(open("src/preprocessing.py", "r", encoding="utf-8").read().splitlines()))
print("models/ dir contents:", os.listdir("models"))

