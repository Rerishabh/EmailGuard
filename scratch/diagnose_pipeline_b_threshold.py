import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

candidate = "pipeline_b_linear_svc"

calib = pd.read_csv("results/stage6c_calibration_oof.csv")
sweep = pd.read_csv("results/stage6c_threshold_sweep.csv")

with open("results/stage6c_best_candidates.json", encoding="utf-8") as f:
    data = json.load(f)

grp = calib[calib["candidate_name"] == candidate]
row = sweep[
    (sweep["candidate_name"] == candidate) &
    (sweep["selected_threshold"].astype(str).str.lower() == "true")
]

assert len(row) == 1, f"Expected one selected sweep row, found {len(row)}"

scores = grp["continuous_score"].to_numpy()
labels = grp["y_true"].to_numpy()

json_tau = float(
    data["candidates"][candidate]["calibration_oof_metrics"]["selected_threshold"]
)
csv_tau = float(row.iloc[0]["threshold"])

def show_metrics(name, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    fpr = fp / (fp + tn) if fp + tn else 0
    print(f"\n{name}")
    print(f"Threshold : {threshold:.17g}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Precision={prec:.9f}")
    print(f"Recall   ={rec:.9f}")
    print(f"F1       ={f1:.9f}")
    print(f"FPR      ={fpr:.9f}")

print("PIPELINE B THRESHOLD PRECISION DIAGNOSTIC")
print("=" * 65)
print(f"JSON threshold : {json_tau:.17g}")
print(f"CSV threshold  : {csv_tau:.17g}")
print(f"Difference     : {abs(json_tau - csv_tau):.17g}")

show_metrics("JSON threshold", json_tau)
show_metrics("CSV threshold", csv_tau)

print("\nScores close to either threshold:")
distance = np.minimum(np.abs(scores - json_tau), np.abs(scores - csv_tau))
nearest = np.argsort(distance)[:10]

for i in nearest:
    print(
        f"score={scores[i]:.17g} | "
        f"label={labels[i]} | "
        f"JSON pred={int(scores[i] >= json_tau)} | "
        f"CSV pred={int(scores[i] >= csv_tau)}"
    )
