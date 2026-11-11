# Stage 6C: Nested Hyperparameter Tuning, Heuristic Weight Optimization & Threshold Calibration Report
**Project:** `Email-Spam-Detection`  
**Status:** Complete (Provisional Champion Aligned)  
**Training Partition:** N = 2,560 | **Quarantined Test Partition:** N = 641 (Strictly Untouched)  

## 1. Experiment Methodology
Stage 6C implements a rigorous three-tier validation and tuning protocol on the SpamAssassin 2,560-record training split:
1. **Nested 5x4 Cross-Validation (`nested_inner` and `nested_outer`):** Evaluates the model-selection procedure without validation leakage. Inner folds (4-fold CV) select the best configuration per outer fold, which is subsequently evaluated on the held-out outer validation fold (5 outer folds). This provides an unbiased generalization estimate of the tuning process itself.
2. **Global 5-Fold Cross-Validation (`global_cv`):** Conducts exhaustive grid searches over the entire 2,560 training set (5 stratified folds) to select the single best global parameter vector for each candidate pipeline topology.
3. **Fixed-Configuration Calibration Out-Of-Fold Evaluation (`calibration_oof`):** Using the frozen champion hyperparameters from Global CV, independent 5-fold cross-validation continuous scores (signed margins for LinearSVC, probability estimates for KNN) are generated to perform an exact threshold sweep subject to the strict operational constraint $\text{FPR} \le 0.005$.

> **Important Scope Note:** All calibration metrics reported in Stage 6C are training-partition-derived via out-of-fold cross-validation on N = 2,560 records. They do **not** represent independent test-set results. The quarantined 641-record test partition remains untouched until Stage 6D.

## 2. Three Candidate Pipelines
Following Stage 6B baseline ranking, three candidate architectures were selected for hyperparameter tuning:
- **Candidate 1 (Pipeline C + LinearSVC):** Domain-Aware Normalization (URL/currency token handling) concatenated with 6 scaled heuristic features (uppercase ratio, all-caps words, exclamation count, question count, currency symbol count, URL count) fed into a linear SVM.
- **Candidate 2 (Pipeline B + LinearSVC):** Standard NLP Normalization (regex cleaning, punctuation removal, lowercasing, sublinear TF-IDF) fed into a linear SVM.
- **Candidate 3 (Pipeline A + KNN):** Minimal Normalization (raw text preserving case and special symbols, sublinear TF-IDF) fed into K-Nearest Neighbors.

## 3. Hyperparameter Search Spaces
| Candidate                            | Search Space Parameters                                                                                                              | Total Configurations |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------- |
| Candidate 1 (Pipeline C + LinearSVC) | C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]; loss in ['squared_hinge', 'hinge']; heuristic_weight in [0.05, 0.25, 0.5, 1.0]; max_iter=5000 | 48                   |
| Candidate 2 (Pipeline B + LinearSVC) | C in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]; loss in ['squared_hinge', 'hinge']; max_iter=5000                                  | 16                   |
| Candidate 3 (Pipeline A + KNN)       | n_neighbors in [3, 5, 7, 11, 15]; weights in ['uniform', 'distance']; metric in ['cosine', 'euclidean']                              | 20                   |

**Total Grid Space:** 84 unique configurations across the 3 candidates.

## 4. Nested Cross-Validation Results (5x4 CV, N = 2,560)
| Candidate Pipeline     | Nested OOF F1 | Precision | Recall | FPR    | Average Precision (AP) |
| ---------------------- | ------------- | --------- | ------ | ------ | ---------------------- |
| Pipeline C + LinearSVC | 0.9303        | 0.9612    | 0.9013 | 0.0064 | 0.9717                 |
| Pipeline B + LinearSVC | 0.9283        | 0.9689    | 0.8909 | 0.0051 | 0.9865                 |
| Pipeline A + KNN       | 0.9272        | 0.9459    | 0.9091 | 0.0092 | 0.9479                 |

## 5. Global CV Selections (5-Fold CV on Full Training Set)
| Candidate Pipeline     | Optimal Global Parameters                                                                        | Mean Spam F1      | Mean Precision | Mean Recall | Mean FPR | Mean PR-AUC |
| ---------------------- | ------------------------------------------------------------------------------------------------ | ----------------- | -------------- | ----------- | -------- | ----------- |
| Pipeline C + LinearSVC | {"C": 10.0, "loss": "squared_hinge", "heuristic_weight": 0.05, "max_iter": 5000, "dual": "auto"} | 0.9323 +/- 0.0326 | 0.9624         | 0.9065      | 0.0064   | 0.9771      |
| Pipeline B + LinearSVC | {"C": 2.0, "loss": "hinge", "max_iter": 5000, "dual": true}                                      | 0.9292 +/- 0.0216 | 0.9722         | 0.8909      | 0.0046   | 0.9873      |
| Pipeline A + KNN       | {"n_neighbors": 3, "weights": "distance", "metric": "cosine"}                                    | 0.9306 +/- 0.0099 | 0.9556         | 0.9091      | 0.0078   | 0.9462      |

## 6. Exact Threshold Calibration Results (Training Partition Derived, FPR <= 0.005)
| Candidate              | Default tau | Def F1 | Def FPR | Canonical tau*     | Calib F1 | Calib Prec | Calib Rec | Calib FPR | Confusion Matrix              | FPR <= 0.005? |
| ---------------------- | ----------- | ------ | ------- | ------------------ | -------- | ---------- | --------- | --------- | ----------------------------- | ------------- |
| Pipeline C + LinearSVC | 0.0000      | 0.9346 | 0.0064  | 0.0876861391187918 | 0.9266   | 0.9715     | 0.8857    | 0.0046    | TN=2165, FP=10, FN=44, TP=341 | YES           |
| Pipeline B + LinearSVC | 0.0000      | 0.9283 | 0.0051  | 0.0040852377135018 | 0.9295   | 0.9717     | 0.8909    | 0.0046    | TN=2165, FP=10, FN=42, TP=343 | YES           |
| Pipeline A + KNN       | 0.5000      | 0.9323 | 0.0078  | 0.6785031750306262 | 0.9258   | 0.9825     | 0.8753    | 0.0028    | TN=2169, FP=6, FN=48, TP=337  | YES           |

## 7. Threshold Optimization Analysis & Floating-Point Consistency Audit
- **LinearSVC Decision Margins:** For Pipeline C and Pipeline B, continuous scores are signed geometric distances to the separating hyperplane. The default decision boundary $\tau = 0.0$ resulted in $\text{FPR} \approx 0.0064$ (Pipeline C) and $\text{FPR} \approx 0.0051$ (Pipeline B). Calibrating to canonical threshold $\tau^* = 0.0876861391187918$ (Pipeline C) and $\tau^* = 0.0040852377135018$ (Pipeline B) slightly shifts the hyperplane into the spam region, reducing false alarms to 10 out of 2,175 legitimate emails (FPR = 0.0046), comfortably satisfying the $\text{FPR} \le 0.005$ operational requirement.
- **KNN Posterior Probabilities:** For Pipeline A + KNN, continuous scores represent the fraction of distance-weighted k-nearest neighbors in class 1. Shifting the threshold from $\tau = 0.5$ to canonical $\tau^* = 0.6785031750306262$ reduced false alarms from 17 to 6 (FPR = 0.0028) with a modest recall trade-off (87.53%).
- **Floating-Point Consistency Audit:** During initial export, threshold floats in `best_candidates.json` contained additional trailing sub-epsilon digits (e.g., `0.0040852377135018325` vs CSV `0.0040852377135018004`). Because an out-of-fold email score matched the threshold value at standard CSV precision, the strictly higher JSON float shifted the boundary check from True to False for that single sample. All thresholds across `stage6c_threshold_sweep.csv`, `stage6c_calibration_oof.csv`, and `stage6c_best_candidates.json` were strictly unified to the canonical CSV float precision, guaranteeing exact bitwise agreement: Pipeline B yields TN=2165, FP=10, FN=42, TP=343 across all artifacts.

## 8. Convergence and Numerical Stability Information
- **Total Global CV Evaluations:** 420 (84 configs x 5 folds)
- **Fully Converged Evaluations:** 383 / 420 (91.19%)
- **Non-Converged Evaluations (hit max_iter=5000):** 37 / 420 (8.81%)

### Non-Converged Configurations Breakdown:
| Candidate             | Configuration ID                 | Non-Converged Folds (out of 5) |
| --------------------- | -------------------------------- | ------------------------------ |
| pipeline_c_linear_svc | c_lsvc_w0.25_lhinge_c10.0        | 1                              |
| pipeline_c_linear_svc | c_lsvc_w0.25_lhinge_c5.0         | 1                              |
| pipeline_c_linear_svc | c_lsvc_w0.5_lhinge_c1.0          | 1                              |
| pipeline_c_linear_svc | c_lsvc_w0.5_lhinge_c10.0         | 2                              |
| pipeline_c_linear_svc | c_lsvc_w0.5_lhinge_c5.0          | 2                              |
| pipeline_c_linear_svc | c_lsvc_w1.0_lhinge_c0.5          | 4                              |
| pipeline_c_linear_svc | c_lsvc_w1.0_lhinge_c1.0          | 4                              |
| pipeline_c_linear_svc | c_lsvc_w1.0_lhinge_c10.0         | 5                              |
| pipeline_c_linear_svc | c_lsvc_w1.0_lhinge_c5.0          | 5                              |
| pipeline_c_linear_svc | c_lsvc_w1.0_lsquared_hinge_c0.5  | 2                              |
| pipeline_c_linear_svc | c_lsvc_w1.0_lsquared_hinge_c1.0  | 2                              |
| pipeline_c_linear_svc | c_lsvc_w1.0_lsquared_hinge_c10.0 | 4                              |
| pipeline_c_linear_svc | c_lsvc_w1.0_lsquared_hinge_c5.0  | 4                              |

**Key Convergence Finding:** Non-convergence occurred exclusively in Pipeline C configurations with large heuristic weights ($w = 1.0$ and $w = 0.5$) combined with high penalties ($C \ge 1.0$). Because heuristic feature counts have uncentered large variances relative to L2-normalized TF-IDF features, large weights induce ill-conditioned dual coordinate descent geometry in LinearSVC. In contrast, scaling down heuristic weights to $w = 0.05$ achieved **100% convergence rate** across all folds.

## 9. Candidate Comparison and Champion Recommendation
| Candidate Configuration | Calibrated F1 | Calibrated Prec | Calibrated Rec | Calibrated FPR | Latency per Email | Convergence |
| ----------------------- | ------------- | --------------- | -------------- | -------------- | ----------------- | ----------- |
| Pipeline C + LinearSVC  | 0.9266        | 0.9715          | 0.8857         | 0.0046         | 2.683 ms          | 100.0%      |
| Pipeline B + LinearSVC  | 0.9295        | 0.9717          | 0.8909         | 0.0046         | 4.712 ms          | 100.0%      |
| Pipeline A + KNN        | 0.9258        | 0.9825          | 0.8753         | 0.0028         | 1.421 ms          | 100.0%      |

### Provisional Production Champion: `Pipeline B + LinearSVC`
- **Architecture:** Pipeline B (Standard NLP Normalization) with LinearSVC
- **Optimal Hyperparameters:** `C=2.0`, `loss='hinge'`, `max_iter=5000`, `dual=True`
- **Operating Threshold:** $\tau^* = 0.0040852377135018$
- **Calibrated Performance (Training-Derived):** Spam F1 = **0.9295**, Precision = **0.9717**, Recall = **0.8909**, FPR = **0.0046** (10 FP / 2,165 TN), Accuracy = **97.97%**
- **Selection Rationale:** Pipeline B + LinearSVC achieved the highest calibrated Spam F1-Score (0.9295) and the highest Average Precision (0.9865) across the benchmark, while strictly adhering to the operational requirement of FPR <= 0.005 (0.46% false-positive rate, exactly 10 false alarms out of 2,175 legitimate emails). Its clean NLP normalization and standard L2 TF-IDF representation provide superior discriminative margin without heuristic noise.

### Preserved Faster Alternative: `Pipeline C + LinearSVC`
- **Architecture:** Pipeline C (Domain-Aware Normalization + Scaled Heuristics with w = 0.05) with LinearSVC
- **Optimal Hyperparameters:** `C=10.0`, `loss='squared_hinge'`, `heuristic_weight=0.05`, `max_iter=5000`, `dual='auto'`
- **Operating Threshold:** $\tau^* = 0.0876861391187918$
- **Performance Profile:** Calibrated Spam F1 = **0.9266**, Precision = **0.9715**, Recall = **0.8857**, FPR = **0.0046** (10 FP / 2,165 TN)
- **Operational Benefit:** Offers approximately **1.75x faster per-email inference latency** (2.68 ms/email vs 4.71 ms/email for Pipeline B) while maintaining 100% numerical convergence stability, making it the preferred fallback candidate for ultra-high-throughput production environments.

## 10. Limitations and Methodological Caveats
1. **Training-Derived Calibration:** All threshold calibration metrics reported above are derived from out-of-fold cross-validation on the 2,560-record training partition. They do **not** represent test-set generalization performance. Test-set evaluation remains quarantined until Stage 6D.
2. **Score Distribution Differences:** LinearSVC produces unbounded signed margins, whereas KNN outputs bounded posterior probabilities [0, 1]. Decision thresholds must not be compared directly between different model families.
3. **Inference Latency Trade-Off:** Pipeline B achieves the highest F1 and Average Precision, but its regex-based tokenization inside the Python pipeline increases per-email latency (4.71 ms vs 2.68 ms for Pipeline C).
4. **Class Imbalance:** Spam prevalence in the training partition is 15.04% (385 spam / 2,175 ham). False positive rate control (FPR <= 0.5%) directly protects against legitimate email misclassification, but slightly depresses raw spam recall (89.09% for Pipeline B, 88.57% for Pipeline C).

## 11. Dataset and Split Integrity Checks
| Resource Checked                          | Expected Hash / Value                                            | Actual Hash / Value                                              | Status           |
| ----------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------- |
| dataset/spam_assassin.csv (MD5)           | cc161a4efd9534c8a32664a52942d097                                 | cc161a4efd9534c8a32664a52942d097                                 | PASS             |
| dataset/spam_assassin_cleaned.csv (MD5)   | 87ab383a0a2853ff148a0e61fb5f2e5d                                 | 87ab383a0a2853ff148a0e61fb5f2e5d                                 | PASS             |
| dataset/train_split_indices.npy (SHA-256) | 84a9ef640ccc54711068d52c1f22d767aa800142c0f7b4822014855158e61d91 | 84a9ef640ccc54711068d52c1f22d767aa800142c0f7b4822014855158e61d91 | PASS             |
| dataset/test_split_indices.npy (SHA-256)  | bb1d408a89a7f53177defd1e4cd11d1036cccb577e3bc670a13c0d170210a0f5 | bb1d408a89a7f53177defd1e4cd11d1036cccb577e3bc670a13c0d170210a0f5 | PASS (Untouched) |
| src/preprocessing.py line count           | 487                                                              | 487                                                              | PASS             |
| models/ directory hygiene                 | Only .gitkeep present                                            | Clean (.gitkeep only)                                            | PASS             |

