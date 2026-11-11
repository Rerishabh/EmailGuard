# Stage 6B: 18-Model Cross-Validation Matrix Results
> [!IMPORTANT]
> All fitting (vectorizers, scalers, classifiers) was confined strictly to
> fold training splits. The held-out test partition (N=641) was never loaded or referenced
> during any fitting or evaluation step in this stage.

## 1. Experimental Methodology
- **Dataset:** `dataset/spam_assassin_cleaned.csv` (3,201 records).
- **Training Partition:** `dataset/train_split_indices.npy` — **N = 2,560** records (80%).
  - Ham (0): 2,175 (84.96%) | Spam (1): 385 (15.04%)
- **Held-Out Test Partition:** `dataset/test_split_indices.npy` — **N = 641** records (20%) [Unused in Stage 6B].
- **Cross-Validation Scheme:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on training partition.
  - Each fold: 2,048 training records, **512 validation records** (Ham: 435, Spam: 77).
- **Execution Matrix:** 3 pipelines × 6 classifiers = **18 combinations × 5 folds = 90 total fold evaluations**.
- **Fault Tolerance:** Atomic per-fold CSV append with `os.fsync` ensuring immediate persistence.
- **Primary Selection Metric:** Mean Spam F1-Score, balancing Precision and Recall under class imbalance.
- **PR-AUC (Average Precision):** Area under Precision-Recall curve resolved dynamically for Spam (class 1).
- **False Positive Rate (FPR):** FP / (FP + TN), evaluating risk of blocking legitimate emails.

### Pipeline Configurations
| Pipeline | Normalization Transformer | Vectorizer | Engineered Heuristics Branch |
|:---|:---|:---|:---|
| **Pipeline A** | `MinimalTextTransformer` (whitespace only) | `TfidfVectorizer(ngram_range=(1,2), min_df=3, sublinear_tf=True)` | None |
| **Pipeline B** | `StandardNLPTextTransformer` (lower, strip punct/digits, stopwords, Porter stem) | Same as A | None |
| **Pipeline C** | `DomainAwareTextTransformer` (placeholders for URL, email, currency, numbers) | Same as A | `EmailHeuristicFeatures` → `StandardScaler(with_mean=False)` |

## 2. 18-Model Ranked Comparison Table

| Rank | Pipeline | Classifier | Spam F1 (Mean ± Std) | Spam Prec (Mean ± Std) | Spam Recall (Mean ± Std) | FPR (Mean ± Std) | PR-AUC (Mean ± Std) | Fit Time (s) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Pipeline C | Linear SVC | 0.9225 ± 0.0224 | 0.9616 ± 0.0188 | 0.8883 ± 0.0491 | 0.0064 ± 0.0034 | 0.9641 ± 0.0240 | 6.684 |
| 2 | Pipeline A | K-Nearest Neighbors | 0.9215 ± 0.0156 | 0.9449 ± 0.0337 | 0.9013 ± 0.0427 | 0.0097 ± 0.0068 | 0.9425 ± 0.0078 | 2.349 |
| 3 | Pipeline B | Linear SVC | 0.9176 ± 0.0248 | 0.9689 ± 0.0203 | 0.8727 ± 0.0463 | 0.0051 ± 0.0034 | 0.9860 ± 0.0070 | 9.956 |
| 4 | Pipeline A | Linear SVC | 0.9117 ± 0.0241 | 0.9681 ± 0.0151 | 0.8623 ± 0.0417 | 0.0051 ± 0.0025 | 0.9847 ± 0.0100 | 2.371 |
| 5 | Pipeline B | K-Nearest Neighbors | 0.9047 ± 0.0096 | 0.9503 ± 0.0269 | 0.8649 ± 0.0385 | 0.0083 ± 0.0050 | 0.9385 ± 0.0243 | 12.360 |
| 6 | Pipeline C | Random Forest | 0.8183 ± 0.0375 | 0.9759 ± 0.0256 | 0.7065 ± 0.0571 | 0.0032 ± 0.0035 | 0.9427 ± 0.0260 | 4.845 |
| 7 | Pipeline A | Logistic Regression | 0.8040 ± 0.0395 | 0.9960 ± 0.0089 | 0.6753 ± 0.0543 | 0.0005 ± 0.0010 | 0.9687 ± 0.0100 | 2.423 |
| 8 | Pipeline A | Random Forest | 0.8035 ± 0.0232 | 0.9817 ± 0.0256 | 0.6805 ± 0.0299 | 0.0023 ± 0.0033 | 0.9357 ± 0.0248 | 2.823 |
| 9 | Pipeline B | Logistic Regression | 0.8007 ± 0.0311 | 0.9960 ± 0.0089 | 0.6701 ± 0.0417 | 0.0005 ± 0.0010 | 0.9715 ± 0.0080 | 9.956 |
| 10 | Pipeline C | Decision Tree | 0.7999 ± 0.0302 | 0.7868 ± 0.0591 | 0.8156 ± 0.0109 | 0.0400 ± 0.0141 | 0.6698 ± 0.0472 | 9.201 |
| 11 | Pipeline A | Decision Tree | 0.7902 ± 0.0303 | 0.8004 ± 0.0484 | 0.7818 ± 0.0360 | 0.0349 ± 0.0097 | 0.6592 ± 0.0451 | 3.979 |
| 12 | Pipeline B | Random Forest | 0.7861 ± 0.0181 | 0.9744 ± 0.0291 | 0.6597 ± 0.0296 | 0.0032 ± 0.0038 | 0.9254 ± 0.0300 | 10.013 |
| 13 | Pipeline C | Logistic Regression | 0.7664 ± 0.0227 | 0.9496 ± 0.0481 | 0.6442 ± 0.0351 | 0.0064 ± 0.0064 | 0.9257 ± 0.0418 | 9.735 |
| 14 | Pipeline B | Decision Tree | 0.7609 ± 0.0390 | 0.7802 ± 0.0480 | 0.7429 ± 0.0336 | 0.0372 ± 0.0091 | 0.6194 ± 0.0555 | 10.655 |
| 15 | Pipeline B | Multinomial NB | 0.5308 ± 0.0665 | 1.0000 ± 0.0000 | 0.3636 ± 0.0656 | 0.0000 ± 0.0000 | 0.9529 ± 0.0101 | 9.818 |
| 16 | Pipeline C | K-Nearest Neighbors | 0.5126 ± 0.0284 | 0.3714 ± 0.0347 | 0.8338 ± 0.0250 | 0.2533 ± 0.0429 | 0.5150 ± 0.0299 | 4.461 |
| 17 | Pipeline A | Multinomial NB | 0.4416 ± 0.0735 | 1.0000 ± 0.0000 | 0.2857 ± 0.0630 | 0.0000 ± 0.0000 | 0.9540 ± 0.0118 | 2.224 |
| 18 | Pipeline C | Multinomial NB | 0.2058 ± 0.0643 | 0.9765 ± 0.0526 | 0.1169 ± 0.0440 | 0.0009 ± 0.0021 | 0.8551 ± 0.0309 | 9.993 |

## 3. In-Depth Architectural Analysis

### 3.1 Preprocessing Strategy Impact (Pipeline A vs B vs C)
1. **Pipeline A (Minimal Normalization):** Retains raw capitalization, punctuation, and explicit domains/URLs. Across all 6 classifiers, Pipeline A achieves a mean Spam F1 of **0.7787** (max: **0.9215**) and mean PR-AUC of **0.9075**. High-signal unigrams like `http`, `click`, `free`, and explicit numbers directly inform linear and tree-based decision boundaries.

2. **Pipeline B (Standard NLP Normalization):** Applying aggressive lowercasing, punctuation stripping, stopword removal, and Porter stemming compresses vocabulary. However, it removes domain-specific artifacts (currency symbols, exclamation bursts, URL structures). It yields mean Spam F1 of **0.7835** (max: **0.9176**) and mean PR-AUC of **0.8989**.

3. **Pipeline C (Domain-Aware + Scaled Heuristics):** Maps specific domain entities to semantic placeholders (`urlplaceholder`, `currencyplaceholder`, `numberplaceholder`) while adding 6 orthogonal structural features (uppercase ratio, all-caps word count, exclamation count, question count, currency count, URL count). Achieves mean Spam F1 of **0.6709** (max: **0.9225**) and mean PR-AUC of **0.8121**.

### 3.2 Heuristic Feature & Scaling Interactions
- **Linear Models (Logistic Regression, LinearSVC):** `StandardScaler(with_mean=False)` scales heuristic variances without shifting zeros, preserving sparse format and ensuring gradient descent treats numerical heuristics and TF-IDF features with comparable weight magnitudes.
- **Multinomial Naive Bayes:** Requires $x_{i,j} \ge 0$. Because `with_mean=False` only divides by $\sigma > 0$ without subtracting the mean $\mu$, all entries remain non-negative, allowing MultinomialNB to train cleanly on the unified CSR matrix.
- **Tree & Ensemble Models (Decision Tree, Random Forest):** Trees split on individual feature thresholds monotonically. The engineered heuristics provide immediate, high-purity early splits on structural traits (e.g., all-caps words, exclamation bursts).

### 3.3 Top Candidate Models Selected for Stage 6C Hyperparameter Tuning

**Candidate 1: Pipeline C + Linear SVC**
- **Spam F1-Score:** 0.9225 ± 0.0224
- **Spam Precision:** 0.9616 ± 0.0188 | **Spam Recall:** 0.8883 ± 0.0491
- **False Positive Rate (FPR):** 0.0064 ± 0.0034
- **PR-AUC (Average Precision):** 0.9641 ± 0.0240
- **Mean Training Time:** 6.684 s
- **Rationale:** Demonstrates top-tier classification balance, extremely low false positive rate on legitimate emails, and high ranking confidence.

**Candidate 2: Pipeline A + K-Nearest Neighbors**
- **Spam F1-Score:** 0.9215 ± 0.0156
- **Spam Precision:** 0.9449 ± 0.0337 | **Spam Recall:** 0.9013 ± 0.0427
- **False Positive Rate (FPR):** 0.0097 ± 0.0068
- **PR-AUC (Average Precision):** 0.9425 ± 0.0078
- **Mean Training Time:** 2.349 s
- **Rationale:** Demonstrates top-tier classification balance, extremely low false positive rate on legitimate emails, and high ranking confidence.

**Candidate 3: Pipeline B + Linear SVC**
- **Spam F1-Score:** 0.9176 ± 0.0248
- **Spam Precision:** 0.9689 ± 0.0203 | **Spam Recall:** 0.8727 ± 0.0463
- **False Positive Rate (FPR):** 0.0051 ± 0.0034
- **PR-AUC (Average Precision):** 0.9860 ± 0.0070
- **Mean Training Time:** 9.956 s
- **Rationale:** Demonstrates top-tier classification balance, extremely low false positive rate on legitimate emails, and high ranking confidence.

## 4. Dataset & Workspace Integrity Verification

| Resource Checked | Expected Hash / Value | Actual Hash / Value | Status |
|:---|:---|:---|:---:|
| `dataset/spam_assassin.csv` MD5 | `cc161a4efd9534c8a32664a52942d097` | `cc161a4efd9534c8a32664a52942d097` | PASS |
| `dataset/spam_assassin_cleaned.csv` MD5 | `87ab383a0a2853ff148a0e61fb5f2e5d` | `87ab383a0a2853ff148a0e61fb5f2e5d` | PASS |
| `dataset/train_split_indices.npy` SHA-256 | `84a9ef640ccc54711068d52c1f22d767aa800142c0f7b4822014855158e61d91` | `84a9ef640ccc54711068d52c1f22d767aa800142c0f7b4822014855158e61d91` | PASS |
| `dataset/test_split_indices.npy` SHA-256 | `bb1d408a89a7f53177defd1e4cd11d1036cccb577e3bc670a13c0d170210a0f5` | `bb1d408a89a7f53177defd1e4cd11d1036cccb577e3bc670a13c0d170210a0f5` | PASS |
| `src/preprocessing.py` line count | 487 lines | 487 lines | PASS |
| `models/` directory contents | `['.gitkeep']` | `['.gitkeep']` | PASS |
