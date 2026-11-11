import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

workspace_dir = Path(__file__).resolve().parent.parent

sweep_path = workspace_dir / "results" / "stage6c_threshold_sweep.csv"
calib_path = workspace_dir / "results" / "stage6c_calibration_oof.csv"
json_path = workspace_dir / "results" / "stage6c_best_candidates.json"
summary_md_path = workspace_dir / "results" / "stage6c_tuning_summary.md"

df_sweep = pd.read_csv(sweep_path)
df_calib = pd.read_csv(calib_path)
with open(json_path, "r", encoding="utf-8") as f:
    best_json = json.load(f)
summary_md_text = summary_md_path.read_text(encoding="utf-8")

candidates = ["pipeline_c_linear_svc", "pipeline_b_linear_svc", "pipeline_a_knn"]

print("=" * 80)
print("INDEPENDENT TRIPLE-ARTIFACT THRESHOLD & METRIC CONSISTENCY AUDIT")
print("=" * 80)

all_agree = True

for cname in candidates:
    print(f"\n--- Candidate: {cname} ---")
    
    # 1. Sweep CSV Canonical
    sel_sweep = df_sweep[(df_sweep["candidate_name"] == cname) & (df_sweep["selected_threshold"].astype(str).str.lower().isin(["true", "1"]))].iloc[0]
    sweep_tau = float(sel_sweep["threshold"])
    sweep_cm = {"tn": int(sel_sweep["tn"]), "fp": int(sel_sweep["fp"]), "fn": int(sel_sweep["fn"]), "tp": int(sel_sweep["tp"])}
    sweep_f1 = float(sel_sweep["f1_spam"])
    sweep_prec = float(sel_sweep["precision_spam"])
    sweep_rec = float(sel_sweep["recall_spam"])
    sweep_fpr = float(sel_sweep["fpr"])
    
    # 2. JSON
    json_tau = float(best_json["candidates"][cname]["calibration_oof_metrics"]["selected_threshold"])
    json_cm = best_json["candidates"][cname]["calibration_oof_metrics"]["confusion_matrix"]
    json_f1 = float(best_json["candidates"][cname]["calibration_oof_metrics"]["calibrated_f1"])
    json_prec = float(best_json["candidates"][cname]["calibration_oof_metrics"]["calibrated_precision"])
    json_rec = float(best_json["candidates"][cname]["calibration_oof_metrics"]["calibrated_recall"])
    json_fpr = float(best_json["candidates"][cname]["calibration_oof_metrics"]["calibrated_fpr"])
    
    # 3. Direct OOF calculation
    c_subset = df_calib[df_calib["candidate_name"] == cname].sort_values("record_index")
    y_true = c_subset["y_true"].to_numpy()
    scores = c_subset["continuous_score"].to_numpy()
    preds = (scores >= sweep_tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    oof_cm = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    oof_f1 = float(f1_score(y_true, preds, zero_division=0))
    oof_prec = float(precision_score(y_true, preds, zero_division=0))
    oof_rec = float(recall_score(y_true, preds, zero_division=0))
    oof_fpr = float(fp / (fp + tn))
    
    # Check threshold match
    tau_match = (sweep_tau == json_tau)
    cm_match = (sweep_cm == json_cm == oof_cm)
    f1_match = np.isclose(sweep_f1, json_f1) and np.isclose(sweep_f1, oof_f1)
    
    print(f"  Canonical Threshold (Sweep CSV): {sweep_tau:.16f}")
    print(f"  JSON Selected Threshold:         {json_tau:.16f} | Match: {tau_match}")
    print(f"  Confusion Matrix (Sweep CSV):    {sweep_cm}")
    print(f"  Confusion Matrix (JSON):         {json_cm}")
    print(f"  Confusion Matrix (OOF Eval):     {oof_cm} | Match: {cm_match}")
    print(f"  Spam F1 (Sweep/JSON/OOF):        {sweep_f1:.6f} / {json_f1:.6f} / {oof_f1:.6f} | Match: {f1_match}")
    print(f"  Precision:                       {oof_prec:.6f} | Recall: {oof_rec:.6f} | FPR: {oof_fpr:.6f}")
    
    if not (tau_match and cm_match and f1_match):
        all_agree = False

print("\n" + "=" * 80)
print(f"OVERALL TRIPLE-ARTIFACT AGREEMENT: {'PASS (ALL THREE AGREE PERFECTLY)' if all_agree else 'FAIL'}")
print("=" * 80)

# Check Champion consistency in JSON
champ_name = best_json["champion_recommendation"]["candidate_name"]
champ_tau = best_json["champion_recommendation"]["selected_threshold"]
champ_cm = best_json["champion_recommendation"]["calibration_metrics"]["confusion_matrix"]
print(f"\nChampion Recommendation: {champ_name}")
print(f"  Champion Threshold: {champ_tau:.16f} (Matches Candidate 1: {champ_tau == best_json['candidates'][champ_name]['calibration_oof_metrics']['selected_threshold']})")
print(f"  Champion CM:        {champ_cm} (Matches Candidate 1: {champ_cm == best_json['candidates'][champ_name]['calibration_oof_metrics']['confusion_matrix']})")

# Integrity Check
def md5(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()
def sha256(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()

print("\n--- Workspace Cryptographic Integrity Attestation ---")
print("spam_assassin.csv MD5:         ", md5(workspace_dir / "dataset" / "spam_assassin.csv"))
print("spam_assassin_cleaned.csv MD5: ", md5(workspace_dir / "dataset" / "spam_assassin_cleaned.csv"))
print("train_split_indices.npy SHA256:", sha256(workspace_dir / "dataset" / "train_split_indices.npy"))
print("test_split_indices.npy SHA256: ", sha256(workspace_dir / "dataset" / "test_split_indices.npy"))
print("src/preprocessing.py lines:    ", len((workspace_dir / "src" / "preprocessing.py").read_text(encoding="utf-8").splitlines()))
print("models/ directory contents:    ", [f.name for f in (workspace_dir / "models").glob("*")])

