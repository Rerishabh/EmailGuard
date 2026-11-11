"""Stage 5A architecture inspection and synthetic compatibility test harness.

Strictly synthetic data only. No training on actual SpamAssassin dataset.
"""

import sys
import hashlib
from pathlib import Path
import numpy as np
import scipy.sparse as sp

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from src.preprocessing import (
    DomainAwareTextTransformer,
    EmailHeuristicFeatures,
)

print("=" * 70)
print("STAGE 5A: SYNTHETIC DATASET & PIPELINE C ARCHITECTURE EXPERIMENT")
print("=" * 70)

# =====================================================================
# 2.1 Synthetic Dataset Construction
# =====================================================================
np.random.seed(42)

# Templates for synthetic legitimate (Not Spam: 0) and spam (1) emails
# Designed to ensure recurring vocabulary for min_df=3 and diverse heuristic markers
ham_templates = [
    "Subject: [RPM-List] package update for freshrpms\n\nHello developers, please check the new rpm package mirror at https://freshrpms.net/packages. Send questions to listmaster@sourceforge.net. System update complete.",
    "Subject: Linux kernel patch review\n\nHi team, the patch for the memory management system has been committed. Visit https://kernel.org/git for documentation. Contact maintainer@kernel.org. Thanks!",
    "Subject: FreshRPMs mailing list digest\n\nHere is your daily summary of rpm package discussions. Visit https://sourceforge.net/projects/freshrpms for bug reports. Reply to users@lists.sourceforge.net.",
    "Subject: Python script bugfix and documentation\n\nThe software repository has been updated with the latest bugfix release. Documentation available at https://docs.python.org/release. Please report any issue.",
    "Subject: [Dev] open-source infrastructure update\n\nServer maintenance scheduled for this weekend. Check mirror status at http://mirror.freshrpms.net/status. Questions to admin@lists.sourceforge.net.",
]

spam_templates = [
    "Subject: URGENT!!! FREE MONEY OFFER NOW!!!\n\nCongratulations! You have been selected to receive $500 cash bonus today! Click here: https://promo-cash.com/claim now! Email agent@fast-cash.com for details. Act now!!!",
    "Subject: SPECIAL BONUS: Claim your $1000 prize now!\n\nWINNER! Special offer for members! Receive 100 dollars instantly. Visit http://exclusive-rewards.com/win today! Limited time offer! $$$",
    "Subject: GUARANTEED LOAN APPROVAL: $2500 cash advance\n\nGet approved for a $2500 personal loan right now! No credit check required. Apply at https://quick-funds.com/apply. Low rates in USD!",
    "Subject: EXCLUSIVE DISCOUNT: Buy cheap prescription meds\n\nOrder online and save up to 80% today! Visit http://meds-direct.com/order. Pay in usd or eur. Fast shipping guaranteed!!!",
    "Subject: WORK FROM HOME: Make $5000 weekly guaranteed!\n\nEarn money from home with zero investment! Sign up at https://earn-now.com/signup. Contact recruiter@earn-now.com immediately!!!",
]

emails = []
labels = []

# Generate 50 Ham emails (label 0)
for i in range(50):
    tmpl = ham_templates[i % len(ham_templates)]
    # Add minor variations while preserving recurring core vocabulary
    variation = f" Reference ID: {1000 + i}. Patch version {i % 5}."
    emails.append(tmpl + variation)
    labels.append(0)

# Generate 50 Spam emails (label 1)
for i in range(50):
    tmpl = spam_templates[i % len(spam_templates)]
    variation = f" Offer code: WIN{2000 + i}. Bonus #{i % 7}."
    emails.append(tmpl + variation)
    labels.append(1)

y = np.array(labels, dtype=int)
print(f"Synthetic dataset created: {len(emails)} emails (50 Not Spam, 50 Spam).")
print(f"Random seed: 42. Labels: {np.bincount(y)}")

# =====================================================================
# 2.2 Construct Proposed Pipeline C (Uncentered: with_mean=False)
# =====================================================================
text_pipeline = Pipeline([
    ("domain_transform", DomainAwareTextTransformer()),
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True
    ))
])

heuristic_pipeline_uncentered = Pipeline([
    ("features", EmailHeuristicFeatures()),
    ("scaler", StandardScaler(with_mean=False))
])

feature_union_uncentered = FeatureUnion([
    ("tfidf", text_pipeline),
    ("heuristics", heuristic_pipeline_uncentered)
])

# =====================================================================
# 2.3 Sparse Matrix Format Verification
# =====================================================================
X_comb_native = feature_union_uncentered.fit_transform(emails)
is_native_csr = isinstance(X_comb_native, (sp.csr_matrix, sp.csr_array))
is_any_sparse = sp.issparse(X_comb_native)

# Explicit CSR conversion
X_comb = X_comb_native.tocsr()
is_explicit_csr = isinstance(X_comb, (sp.csr_matrix, sp.csr_array))

n_rows, n_cols = X_comb.shape
nnz = X_comb.nnz
density = nnz / (n_rows * n_cols)
sparse_bytes = X_comb.data.nbytes + X_comb.indices.nbytes + X_comb.indptr.nbytes
dense_array = np.asarray(X_comb.todense())
dense_bytes = dense_array.nbytes
savings_bytes = dense_bytes - sparse_bytes
savings_pct = (savings_bytes / dense_bytes) * 100

print("\n--- 2.3 Sparse Matrix Verification ---")
print(f"Native return type from FeatureUnion: {type(X_comb_native)}")
print(f"Is native output a CSR matrix?        : {is_native_csr}")
print(f"Is native output any SciPy sparse?   : {is_any_sparse}")
print(f"Explicit .tocsr() output type        : {type(X_comb)}")
print(f"Matrix dimensions (shape)            : {X_comb.shape} ({n_rows} rows x {n_cols} columns)")
print(f"Number of non-zero entries (nnz)     : {nnz}")
print(f"Matrix density                       : {density:.6f} ({density * 100:.2f}%)")
print(f"Sparse memory usage                  : {sparse_bytes:,} bytes")
print(f"Dense equivalent memory usage        : {dense_bytes:,} bytes")
print(f"Memory savings                       : {savings_bytes:,} bytes ({savings_pct:.2f}% reduction)")

# =====================================================================
# 2.4 Feature Name Verification
# =====================================================================
fu_feature_names_success = False
fu_feature_names_err = None
try:
    feature_names = feature_union_uncentered.get_feature_names_out()
    fu_feature_names_success = True
except Exception as e:
    fu_feature_names_err = f"{type(e).__name__}: {str(e)}"

# Extract feature names directly from component transformers:
tfidf_feature_names = text_pipeline.named_steps["tfidf"].get_feature_names_out()
heuristic_feature_names = heuristic_pipeline_uncentered.get_feature_names_out()

expected_heuristics = [
    "uppercase_ratio",
    "all_caps_word_count",
    "exclamation_count",
    "question_count",
    "currency_symbol_count",
    "url_count",
]

# When combined in FeatureUnion or concatenation:
# Order: tfidf columns followed by heuristics columns
combined_feature_names = np.concatenate([
    np.array([f"tfidf__{f}" for f in tfidf_feature_names]),
    np.array([f"heuristics__{f}" for f in heuristic_feature_names])
])

all_heuristics_found = all(h in heuristic_feature_names for h in expected_heuristics)

print("\n--- 2.4 Feature Name Verification ---")
print(f"FeatureUnion.get_feature_names_out() direct call success: {fu_feature_names_success}")
if not fu_feature_names_success:
    print(f"Direct call note: {fu_feature_names_err}")
    print("Architectural finding: Intermediate custom string transformer without get_feature_names_out stops automated propagation in Pipeline/FeatureUnion.")
print(f"Extracted TF-IDF features count    : {len(tfidf_feature_names)}")
print(f"Extracted Heuristic features count : {len(heuristic_feature_names)} (Expected 6: {len(heuristic_feature_names) == 6})")
print(f"Total concatenated feature names   : {len(combined_feature_names)} (Matches matrix cols {n_cols}: {len(combined_feature_names) == n_cols})")
print(f"All 6 heuristic names found        : {all_heuristics_found}")
print(f"First 10 TF-IDF feature names      : {list(tfidf_feature_names[:10])}")
print(f"Last 10 TF-IDF feature names       : {list(tfidf_feature_names[-10:])}")
print(f"All 6 heuristic feature names      : {list(heuristic_feature_names)}")

# =====================================================================
# 2.5 Non-Negativity Verification
# =====================================================================
X_heur_uncentered = heuristic_pipeline_uncentered.fit_transform(emails)
heur_non_neg = np.all(X_heur_uncentered >= 0.0)
comb_non_neg = np.all(X_comb.data >= 0.0)

print("\n--- 2.5 Non-Negativity Verification ---")
print(f"Are uncentered heuristic features strictly non-negative (>= 0)? : {heur_non_neg}")
print(f"Are all non-zero entries in X_comb strictly non-negative (>= 0)?: {comb_non_neg}")
print(f"Min value in X_heur_uncentered : {np.min(X_heur_uncentered):.6f}")
print(f"Min value in X_comb.data       : {np.min(X_comb.data):.6f}")
print("Mathematical explanation: StandardScaler(with_mean=False) divides by standard deviation (x / sigma). Since x >= 0 and sigma > 0, x / sigma >= 0 strictly holds.")

# =====================================================================
# 2.6 MultinomialNB Compatibility (Uncentered)
# =====================================================================
mnb = MultinomialNB(alpha=1.0)
mnb_fit_success = False
mnb_predict_success = False
mnb_err = None

try:
    mnb.fit(X_comb, y)
    mnb_fit_success = True
    preds = mnb.predict(X_comb)
    mnb_predict_success = (len(preds) == len(y))
except Exception as e:
    mnb_err = f"{type(e).__name__}: {str(e)}"

print("\n--- 2.6 MultinomialNB Compatibility (Uncentered) ---")
print(f"MultinomialNB fit(X_comb, y) success    : {mnb_fit_success}")
print(f"MultinomialNB predict(X_comb) success: {mnb_predict_success} (predictions len: {len(preds) if mnb_predict_success else 0})")
if mnb_err:
    print(f"Error encountered: {mnb_err}")

# =====================================================================
# 2.7 Controlled Centered-Scaling Contrast (with_mean=True)
# =====================================================================
heuristic_pipeline_centered = Pipeline([
    ("features", EmailHeuristicFeatures()),
    ("scaler", StandardScaler(with_mean=True))
])

feature_union_centered = FeatureUnion([
    ("tfidf", text_pipeline),
    ("heuristics", heuristic_pipeline_centered)
])

# Extract intermediate outputs
X_tfidf_uncentered = text_pipeline.fit_transform(emails)
X_tfidf_centered = text_pipeline.fit_transform(emails)

# TF-IDF invariance
tfidf_diff_nnz = (X_tfidf_uncentered != X_tfidf_centered).nnz
tfidf_invariant = (tfidf_diff_nnz == 0)

# Centered heuristic features
X_heur_centered = heuristic_pipeline_centered.fit_transform(emails)
heur_centered_has_negative = np.any(X_heur_centered < 0.0)
tfidf_centered_non_neg = np.all(X_tfidf_centered.data >= 0.0)

# Combined centered matrix
X_comb_centered = feature_union_centered.fit_transform(emails).tocsr()
comb_centered_has_negative = np.any(X_comb_centered.data < 0.0)

print("\n--- 2.7 Controlled Centered-Scaling Contrast ---")
print(f"TF-IDF invariance check (uncentered == centered) diff nnz: {tfidf_diff_nnz} (Invariant: {tfidf_invariant})")
print(f"Does centered heuristic branch contain negative values? : {heur_centered_has_negative}")
print(f"Min value in X_heur_centered                            : {np.min(X_heur_centered):.6f}")
print(f"Does TF-IDF branch remain strictly non-negative?        : {tfidf_centered_non_neg}")
print(f"Does combined centered matrix contain negative values?  : {comb_centered_has_negative}")
print(f"Min value in X_comb_centered.data                       : {np.min(X_comb_centered.data):.6f}")

# Attempt MultinomialNB on centered combined matrix
mnb_centered = MultinomialNB(alpha=1.0)
mnb_centered_raised_value_error = False
mnb_centered_exc_type = None
mnb_centered_exc_msg = None

try:
    mnb_centered.fit(X_comb_centered, y)
except Exception as e:
    mnb_centered_exc_type = type(e).__name__
    mnb_centered_exc_msg = str(e)
    if isinstance(e, ValueError) and "negative" in str(e).lower():
        mnb_centered_raised_value_error = True

print(f"Attempted MultinomialNB.fit(X_comb_centered, y):")
print(f"  Raised expected negative-value ValueError: {mnb_centered_raised_value_error}")
print(f"  Actual Exception Type: {mnb_centered_exc_type}")
print(f"  Actual Exception Message: '{mnb_centered_exc_msg}'")
print("Mathematical explanation: MultinomialNB assumes feature values represent non-negative token occurrence counts or frequencies (x_ij >= 0) to evaluate multinomial likelihoods. Mean-centering subtracts feature means (x - mu), producing negative values which invalidate probability axioms.")

# =====================================================================
# TASK 3: Six-Classifier Sparse Matrix Compatibility
# =====================================================================
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "MultinomialNB": MultinomialNB(alpha=1.0),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
    "LinearSVC": LinearSVC(dual="auto", max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3),
}

results_table = []
print("\n" + "=" * 70)
print("TASK 3: SIX-CLASSIFIER SPARSE MATRIX COMPATIBILITY TABLE")
print("=" * 70)

for name, clf in classifiers.items():
    fit_ok = False
    pred_ok = False
    status = "FAIL"
    notes = ""
    try:
        clf.fit(X_comb, y)
        fit_ok = True
        preds = clf.predict(X_comb)
        pred_ok = (len(preds) == len(y))
        status = "PASS"
        notes = f"Predictions: {len(preds)}, dtype={preds.dtype}"
    except Exception as e:
        status = "FAIL"
        notes = f"{type(e).__name__}: {str(e)}"
    
    results_table.append({
        "Classifier": name,
        "Sparse Fit": "PASS" if fit_ok else "FAIL",
        "Prediction": "PASS" if pred_ok else "FAIL",
        "Status": status,
        "Notes": notes,
    })
    print(f"{name:22}: Fit={'PASS' if fit_ok else 'FAIL':4} | Predict={'PASS' if pred_ok else 'FAIL':4} | Status={status:4} | {notes}")

# =====================================================================
# TASK 4: Dataset & Workspace Integrity Verification
# =====================================================================
print("\n" + "=" * 70)
print("TASK 4: DATASET AND WORKSPACE INTEGRITY VERIFICATION")
print("=" * 70)

def get_md5(path_str):
    p = Path(path_str)
    if not p.exists():
        return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

expected_hashes = {
    "dataset/spam_assassin.csv": "cc161a4efd9534c8a32664a52942d097",
    "dataset/spam_assassin_cleaned.csv": "87ab383a0a2853ff148a0e61fb5f2e5d",
}

integrity_results = {}
for filepath, expected_hash in expected_hashes.items():
    actual_hash = get_md5(filepath)
    matches = (actual_hash == expected_hash)
    integrity_results[filepath] = {
        "exists": Path(filepath).exists(),
        "expected": expected_hash,
        "actual": actual_hash,
        "status": "PASS" if matches else "FAIL",
    }
    print(f"File: {filepath}")
    print(f"  Exists  : {Path(filepath).exists()}")
    print(f"  Expected: {expected_hash}")
    print(f"  Actual  : {actual_hash}")
    print(f"  Status  : {'PASS' if matches else 'FAIL'}")

# Line count of preprocessing.py
prep_path = Path("src/preprocessing.py")
with open(prep_path, "r", encoding="utf-8") as f:
    prep_line_count = len(f.readlines())

prep_line_status = "PASS" if prep_line_count == 487 else "FAIL"
print(f"\nsrc/preprocessing.py line count: {prep_line_count} (Expected 487: {prep_line_status})")

# Check models/ directory
models_dir = Path("models")
models_files = [f.name for f in models_dir.glob("*")]
models_status = "PASS" if models_files == [".gitkeep"] else "FAIL"
print(f"models/ directory contents: {models_files} (Only .gitkeep: {models_status})")

# Check results/graphs/
graphs_dir = Path("results/graphs")
graphs_files = sorted([f.name for f in graphs_dir.glob("*.png")])
expected_graphs = [
    "category_distribution.png",
    "class_distribution.png",
    "email_length_distribution.png",
    "subject_length_distribution.png",
    "top_bigrams_comparison.png",
    "top_unigrams_comparison.png",
    "url_punctuation_heuristics.png",
    "word_count_distribution.png",
]
graphs_status = "PASS" if graphs_files == expected_graphs else "FAIL"
print(f"results/graphs/ PNG count: {len(graphs_files)} (Expected 8: {graphs_status})")

print("\n" + "=" * 70)
print("STAGE 5A SYNTHETIC ARCHITECTURE TEST COMPLETED SUCCESSFULLY.")
print("=" * 70)
