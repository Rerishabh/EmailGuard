# EmailGuard — Intelligent Email Spam Detection Using NLP

**EmailGuard** is a machine learning and Natural Language Processing (NLP) project for binary email classification (**Ham / Not Spam** vs. **Spam**). It combines RFC-822 MIME email parsing, comparative multi-pipeline benchmarking under a strict False Positive Rate constraint, and an interactive **Streamlit** application for live threat inspection and evaluation visualization.

<p align="center">
  <a href="https://emailguard-rishabh.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-EmailGuard-brightgreen?style=for-the-badge" alt="Live Demo">
  </a>
</p>

<p align="center">
  <img src="docs/assets/banner.png"
       alt="EmailGuard - Intelligent Email Spam Detection"
       width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/ML-Scikit--learn-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/NLP-NLTK-4B8BBE">
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/Dataset-SpamAssassin-6C8EBF">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Classifier-LinearSVC-blue">
  <img src="https://img.shields.io/badge/Test_Accuracy-99.06%25-success">
  <img src="https://img.shields.io/badge/False_Positive_Rate-0.37%25-brightgreen">
  <img src="https://img.shields.io/badge/Status-Completed-success">
</p>

---
## 🚀 Live Demo

Try EmailGuard here:

**[Launch EmailGuard](https://emailguard-rishabh.streamlit.app/)**


The Streamlit dashboard (`app/app.py`) organizes inspection and empirical evaluation into four interactive tabs:

- **Overview & Live Analyzer:** Real-time email classification using audited Ham/Spam presets or custom Subject/Body input, displaying decision scores, operating thresholds, boundary deltas, single-email latency, decision margin SVG gauges, Pipeline C structural heuristics, and corpus summary metrics.
- **6-Model Benchmark:** Interactive 5-fold cross-validation tables and comparative charts across all evaluated pipelines and classifiers.
- **Confusion Matrices & Evaluation:** Held-out test partition metrics (`N = 641`), confusion matrix heatmap, and Precision-Recall curve.
- **Architecture & Methodology:** End-to-end NLP workflow and side-by-side comparison of Pipelines A, B, and C.

---

## Dataset & Attribution

This benchmark uses the historical **SpamAssassin Public Corpus (2002–2003)**:

- **Official Source:** [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/)
- **Archives Used:** `easy_ham`, `hard_ham`, and `spam`

All original email messages are credited to the SpamAssassin Public Corpus project. After RFC-822 MIME parsing and full-text deduplication:

| Metric | Count |
| :--- | :---: |
| Original parsed emails | 3,250 |
| Exact duplicate records removed | 49 |
| Unique emails after deduplication | 3,201 |
| Not Spam (`Ham`, label `0`) | 2,720 |
| Spam (label `1`) | 481 |
| Stratified Training Partition (80%) | 2,560 |
| Held-Out Test Partition (20%) | 641 |

> **Historical Benchmark Note:** The SpamAssassin corpus reflects early-2000s email distributions. Benchmark results on this archive should not be interpreted as evidence of effectiveness against modern adversarial spam or contemporary phishing threats.

---

## Key Features

- Automated SpamAssassin archive download, RFC-822 MIME parsing, and attachment stripping (`src/parse_corpus.py`, `src/preprocessing.py`)
- Exact full-text deduplication and strictly disjoint 80/20 stratified train/test splitting
- Three modular Scikit-learn feature engineering pipelines (`Pipeline A`, `Pipeline B`, `Pipeline C`)
- 5-fold stratified cross-validation across six distinct machine learning classifiers
- Out-of-fold hyperparameter tuning and False Positive Rate calibration (`FPR <= 0.50%`)
- Configurable operating threshold toggle in Streamlit (Calibrated threshold vs. `0.0` default boundary)
- Continuous signed decision margin visualization and live inference latency measurement

---

## Machine Learning Methodology

1. **Leak-Free Data Partitioning:** The 3,201 deduplicated emails were stratified into 2,560 training records and 641 held-out test records (`dataset/train_split_indices.npy`, `dataset/test_split_indices.npy`), with test records kept isolated until final single-pass evaluation.
2. **Predefined Model Selection Protocol:** Candidate pipelines and classifiers were benchmarked using 5-fold stratified cross-validation on the training partition (`N = 2,560`), followed by nested cross-validation hyperparameter tuning and out-of-fold threshold calibration targeting **False Positive Rate <= 0.50%** to prioritize protecting legitimate mail.
3. **Champion Selection:** **Pipeline B + LinearSVC** was officially selected as the **Official Benchmark Champion** via the predefined training and cross-validation protocol—**not** based on held-out test performance. **Pipeline C + LinearSVC** is provided in the application as an alternative with lower inference latency.
4. **Decision Score & Threshold Contract:** `LinearSVC.decision_function()` outputs a **continuous signed decision score (decision margin)** relative to the separating hyperplane, **not a probability or percentage confidence**. Classification compares the score against the active threshold (`margin >= threshold` triggers Spam):
   - **Default decision boundary:** `0.0`
   - **Calibrated operating threshold (Pipeline B):** `~0.0041` (`0.0040852377135018`)
   - **Calibrated operating threshold (Pipeline C):** `~0.0877` (`0.0876861391187918`)

---

## Feature Processing Pipelines (A, B, C)

All pipelines vectorize normalized text using sublinear TF-IDF (`ngram_range=(1, 2)`, `min_df=3`, `sublinear_tf=True`):

- **Pipeline A (Minimal Preprocessing):** Whitespace and line-ending normalization while preserving original casing, punctuation, symbols, and URLs, followed by sublinear TF-IDF.
- **Pipeline B (Standard NLP — Official Benchmark Champion Topology):** Lowercasing, punctuation and numeric stripping, NLTK English stopword removal, and Porter stemming, followed by sublinear TF-IDF.
- **Pipeline C (Domain-Aware + Heuristics):** Replaces URLs, email addresses, currency amounts, and numeric values with placeholder tokens for TF-IDF, combined via `FeatureUnion` with 6 `StandardScaler`-normalized structural heuristic features (uppercase character ratio, ALL-CAPS word count, exclamation marks, question marks, currency symbols, and URL count) weighted at `0.05`.

---

## Evaluated Classifiers

Six supervised classifiers (`src/pipeline.py`) were trained and compared across all three pipelines during cross-validation:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Decision Tree
4. Random Forest
5. Linear Support Vector Classifier (`LinearSVC`)
6. K-Nearest Neighbors (`KNN`)

---

## Model Performance

Final single-pass held-out test results (`N = 641`: 545 Ham, 96 Spam) for the **Official Benchmark Champion** (**Pipeline B + LinearSVC**, `C=2.0`, `loss="hinge"`, `dual=True`, `max_iter=5000`) at the calibrated operating threshold (`0.0040852377135018`):

| Evaluation Metric | Held-Out Test Result (`N = 641`) |
| :--- | :---: |
| Accuracy | 99.064% |
| Spam Precision | 97.872% |
| Spam Recall | 95.833% |
| Spam F1-Score | 0.96842 |
| False Positive Rate (FPR) | 0.367% |
| Not Spam Precision | 99.269% |
| Not Spam Recall | 99.633% |
| Average Precision (AP) | 0.997796 |

For latency-sensitive workloads, **Pipeline C + LinearSVC** serves as an in-app alternative averaging `2.02 ms/email` (`~2.72x` faster than Pipeline B's `5.48 ms/email` benchmarked batch per-email latency). Average Precision (`AP`) is computed directly via `sklearn.metrics.average_precision_score(y_test, scores)` over the continuous `LinearSVC` decision scores.

*Note: These results come strictly from the isolated held-out split of the historical SpamAssassin benchmark and are not a guarantee of production filtering performance.*

---

## Held-Out Test Confusion Matrix

**Official Benchmark Champion** (**Pipeline B + LinearSVC**, calibrated threshold) predictions across the 641 held-out test emails:

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 543 | 2 |
| Spam | 4 | 92 |

---

## Technology Stack

Defined in `requirements.txt` using minimum version bounds (`>=`):

- **Language:** Python
- **Machine Learning:** Scikit-learn (`>= 1.3.0`)
- **NLP:** NLTK (`>= 3.8.0`)
- **Data Processing:** Pandas (`>= 2.0.0`), NumPy (`>= 1.24.0`)
- **Web Interface:** Streamlit (`>= 1.28.0`)
- **Visualization:** Matplotlib (`>= 3.7.0`), Seaborn (`>= 0.12.0`)
- **Model Serialization:** Joblib (`>= 1.3.0`)

---

## Project Structure

```text
Email-Spam-Detection/
├── .streamlit/         # Light Warm Minimalist theme configuration
├── app/                # Streamlit UI (app.py), components, custom CSS, and model loader
├── dataset/            # CSV datasets and raw archives (gitignored); .npy train/test split indices (not in .gitignore; Git tracking unverified because Git is not initialized here)
├── models/             # Serialized champion .joblib (gitignored) and champion_config.json
├── notebooks/          # Exploratory data analysis and baseline experimentation notebook
├── results/            # Cross-validation summaries, tuning configs, and evaluation reports
├── scratch/            # Automated verification harnesses and evaluation scripts
├── src/                # Corpus parser, preprocessing transformers, pipelines, train/tune modules
├── .gitignore          # Excludes local CSV datasets, archives, .joblib/.pkl models, and venvs
├── requirements.txt    # Minimum package dependency specifications
└── README.md           # Project documentation
```

---

## Installation & Setup (Windows)

1. **Clone the repository and navigate into the project directory:**

   ```powershell
   git clone https://github.com/Rerishabh/EmailGuard.git
   cd EmailGuard
   ```

   *(If working inside an existing local directory named `Email-Spam-Detection`, run `cd Email-Spam-Detection` instead.)*

2. **Create and activate a Python virtual environment:**

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies and download NLTK stopwords:**

   ```powershell
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('stopwords')"
   ```

4. **Launch the Streamlit application:**

   ```powershell
   streamlit run app/app.py
   ```

   Open [http://localhost:8501](http://localhost:8501) in your browser once the server starts.

---

## Artifact Integrity & Local Model Regeneration

- **Gitignored Artifacts & Split Indices Status:** The Streamlit application loads `models/champion_pipeline.joblib` (and fits Pipeline C in memory from `dataset/spam_assassin_cleaned.csv` and `dataset/train_split_indices.npy` when selected). In `.gitignore`, `dataset/*.csv`, `dataset/archives/`, and `models/*.joblib` (`*.pkl`) are excluded from version control, while `dataset/*.npy` is not listed in `.gitignore`. However, because Git is not initialized in this workspace, the Git tracking status of `dataset/train_split_indices.npy` and `dataset/test_split_indices.npy` cannot currently be verified from local repository metadata.
- **Verification of Training & Pipeline Helpers:**
  - `src/train.py` is an importable cross-validation library module (`run_cross_validation_single`, `run_full_cv_matrix`), **not** a CLI script—commands such as `python src/train.py --pipeline B --classifier linear_svc` are not implemented.
  - `src/pipeline.py` exports `create_pipeline_b(classifier=..., vectorizer_params=...)` and `get_classifier(...)`—no `build_pipeline_b()` function exists in the codebase. Fitting `create_pipeline_b()` without explicitly passing `get_classifier("linear_svc", C=2.0, loss="hinge", dual=True, max_iter=5000)` produces an unclassified vectorizer pipeline without `LinearSVC`.
- **Verified Local Regeneration Procedure:**
  1. Download and parse the SpamAssassin archives into `dataset/spam_assassin.csv` (3,250 records):
     ```powershell
     python src/parse_corpus.py
     ```
  2. Clean and deduplicate full-text records into `dataset/spam_assassin_cleaned.csv` (3,201 unique emails with columns `text` and `label`):
     ```powershell
     python -c "from src.preprocessing import clean_dataset; clean_dataset('dataset/spam_assassin.csv', 'dataset/spam_assassin_cleaned.csv')"
     ```
  3. Fit and serialize the **Official Benchmark Champion** pipeline (`models/champion_pipeline.joblib`) strictly on `dataset/train_split_indices.npy`, ensuring `models/` exists first:
     ```powershell
     python -c "from pathlib import Path; import joblib, numpy as np, pandas as pd; from src.pipeline import create_pipeline_b, get_classifier; df = pd.read_csv('dataset/spam_assassin_cleaned.csv'); train_idx = np.load('dataset/train_split_indices.npy'); clf = get_classifier('linear_svc', C=2.0, loss='hinge', dual=True, max_iter=5000); pipe = create_pipeline_b(classifier=clf); pipe.fit(df.iloc[train_idx]['text'].to_numpy(), df.iloc[train_idx]['label'].to_numpy().astype(int)); Path('models').mkdir(parents=True, exist_ok=True); joblib.dump(pipe, 'models/champion_pipeline.joblib')"
     ```
     *Note:* This command reproduces and serializes only the fitted scikit-learn pipeline (`models/champion_pipeline.joblib`). The calibrated operating threshold (`0.0040852377135018`) originates from nested cross-validation calibration (`results/stage6c_best_candidates.json`), and `models/champion_config.json` is generated by the Stage 6D evaluation script described below.
- **Stage 6D Evaluation Script (`scratch/test_stage6d_evaluation.py`) Behavior & Preconditions:**
  - **Preconditions:** Requires `dataset/spam_assassin.csv`, `dataset/spam_assassin_cleaned.csv`, `dataset/train_split_indices.npy`, and `dataset/test_split_indices.npy` with matching MD5/SHA-256 checksums, `src/preprocessing.py` intact (487 lines), and `results/stage6c_best_candidates.json` present.
  - **Model Artifact Protection (No Automatic Backup / No Overwrite of Existing Models):** Before fitting, `scratch/test_stage6d_evaluation.py` asserts that `models/` contains **no files other than `.gitkeep`**. If `models/champion_pipeline.joblib` or `models/champion_config.json` already exists, the script aborts with an `AssertionError`. It does **not** back up existing model files. When `models/` is empty (except `.gitkeep`), it writes `models/champion_pipeline.joblib` and `models/champion_config.json` and overwrites the Stage 6D result files in `results/` (`stage6d_test_predictions.csv`, `stage6d_evaluation_summary.csv`, `stage6d_evaluation_summary.md`).

---

## Limitations & Operational Boundaries

- **Historical Dataset Scope:** The corpus captures email patterns from 2002–2003 and does not adequately represent modern phishing techniques, adversarial homoglyphs, image/PDF obfuscation, or LLM-generated lures.
- **Content-Only Scope:** Classification operates strictly on email text and structural heuristics; it does **not** replace SPF, DKIM, DMARC, sender/IP reputation, attachment sandboxing, or a complete enterprise email security gateway.
- **Threshold Sensitivity:** Operating thresholds (`~0.0041` for Pipeline B, `~0.0877` for Pipeline C) were calibrated on the SpamAssassin training distribution and may require recalibration for different real-world email distributions.
- **Academic Demonstration Status:** Built as a reproducible machine learning benchmark and demonstration prototype rather than a production-grade email security service.

---

## Objective & GitHub Topics

**Objective:** Deliver a leak-free, reproducible NLP spam classification study—combining multi-pipeline cross-validation, FPR-constrained decision calibration, and an interactive Streamlit diagnostic dashboard—for academic benchmarking and software engineering demonstration.

**GitHub Topics:** `machine-learning` `nlp` `email-spam-detection` `python` `streamlit`
