"""Comprehensive test harness for Stage 5B pipeline factory implementation.

Verifies:
1. Factory imports, pipeline instantiations (with/without classifiers), and cloning via sklearn.base.clone.
2. Parameter propagation (custom min_df, C, max_depth, etc.).
3. Sparse matrix format (issparse and strictly csr_matrix), non-negativity (Pipeline C),
   un-fitted exception handling, feature name extraction alignment, and heuristic branch prefixing.
4. Fit & predict across all six supported classifiers with Pipeline C on synthetic data.
5. Workspace and dataset integrity checks (MD5 checksums, line counts, models/ hygiene).
"""

import hashlib
from pathlib import Path
import sys
from typing import List

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.base import clone
from sklearn.pipeline import FeatureUnion, Pipeline

# Ensure workspace root is in sys.path
workspace_dir = Path(__file__).resolve().parent.parent
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

from src.pipeline import (
    _ensure_csr,
    build_text_vectorizer,
    create_pipeline_a,
    create_pipeline_b,
    create_pipeline_c,
    extract_feature_names,
    get_classifier,
)


def generate_synthetic_emails() -> tuple[List[str], np.ndarray]:
    """Generate 20 synthetic emails (10 ham, 10 spam) with tokens repeating >= 3 times."""
    ham_emails = [
        "Hi team, please review the quarterly budget report before the morning meeting tomorrow.",
        "Team, the quarterly project roadmap documentation has been updated for our review meeting.",
        "Please find the quarterly financial budget summary attached. Let the team know if any review is needed.",
        "Good morning team, let us review the software project status update during today's meeting.",
        "Team meeting reminder: quarterly project evaluation and technical report review tomorrow morning.",
        "Can the team review the final quarterly documentation before our scheduled meeting tomorrow?",
        "Please review the attached project schedule report for the upcoming quarterly planning meeting.",
        "Team discussion on the project roadmap update and quarterly engineering report at 10 AM meeting.",
        "Morning everyone, please check the quarterly budget report and send project notes for the meeting.",
        "Quarterly project review meeting is scheduled for tomorrow. Please review all budget report items.",
    ]
    
    spam_emails = [
        "URGENT WINNER: Claim your FREE $500 cash prize bonus now! Click http://reward.example.com to win!",
        "CONGRATULATIONS: You won a FREE $1000 cash reward prize! Visit http://winner.example.com immediately!",
        "CLAIM CASH NOW: Exclusive $250 prize bonus waiting for you! Click http://cash-bonus.example.com today!",
        "ACT FAST: Claim your FREE prize cash payout of $750 today! Click http://promo.example.com now!",
        "SPECIAL OFFER: Free $500 bonus cash reward! Visit http://gift.example.com to claim your prize now!",
        "URGENT ALERT: You have a FREE $1500 prize reward voucher! Click http://claim.example.com to win cash!",
        "FINAL NOTICE: Claim your $300 cash prize bonus today! Visit http://payout.example.com for free reward!",
        "EXCLUSIVE: Free cash payout of $2000! Click http://jackpot.example.com now to claim your cash prize!",
        "CONGRATULATIONS: Your free $1200 prize reward is ready! Visit http://rewards.example.com to claim cash!",
        "WIN BIG TODAY: Claim your FREE $500 cash bonus prize! Click http://instant-cash.example.com right now!",
    ]
    
    emails = ham_emails + spam_emails
    labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    return emails, labels


def test_1_factory_import_and_cloning():
    print("\n" + "=" * 70)
    print("TEST 1: FACTORY IMPORT & CLONING (sklearn.base.clone)")
    print("=" * 70)

    # Instantiate without classifiers
    pipe_a_feat = create_pipeline_a()
    pipe_b_feat = create_pipeline_b()
    pipe_c_feat = create_pipeline_c()

    # Instantiate with classifiers
    clf_lr = get_classifier("logistic_regression")
    clf_mnb = get_classifier("multinomial_nb")
    clf_rf = get_classifier("random_forest")

    pipe_a_clf = create_pipeline_a(classifier=clf_lr)
    pipe_b_clf = create_pipeline_b(classifier=clf_mnb)
    pipe_c_clf = create_pipeline_c(classifier=clf_rf)

    all_pipelines = {
        "Pipeline A (feature-only)": pipe_a_feat,
        "Pipeline B (feature-only)": pipe_b_feat,
        "Pipeline C (feature-only)": pipe_c_feat,
        "Pipeline A (with LogisticRegression)": pipe_a_clf,
        "Pipeline B (with MultinomialNB)": pipe_b_clf,
        "Pipeline C (with RandomForest)": pipe_c_clf,
    }

    for name, pipe in all_pipelines.items():
        assert isinstance(pipe, Pipeline), f"{name} is not an instance of Pipeline"
        cloned_pipe = clone(pipe)
        assert cloned_pipe is not pipe, f"{name} clone returned identical reference"
        assert len(cloned_pipe.steps) == len(pipe.steps), f"{name} clone step count mismatch"
        print(f"  [PASS] {name:<38} successfully instantiated and cloned.")


def test_2_parameter_propagation():
    print("\n" + "=" * 70)
    print("TEST 2: PARAMETER PROPAGATION (Custom Vectorizer & Classifier Params)")
    print("=" * 70)

    # Custom vectorizer params
    custom_vec_params = {
        "min_df": 5,
        "ngram_range": (1, 3),
        "sublinear_tf": False,
        "max_features": 1000,
    }

    # Custom classifier kwargs
    clf_custom = get_classifier("logistic_regression", C=0.5, max_iter=500, penalty="l2")
    assert clf_custom.C == 0.5, f"Expected C=0.5, got {clf_custom.C}"
    assert clf_custom.max_iter == 500, f"Expected max_iter=500, got {clf_custom.max_iter}"
    print("  [PASS] get_classifier('logistic_regression', C=0.5, max_iter=500) overrides confirmed.")

    # Test unknown classifier raises ValueError
    try:
        get_classifier("invalid_classifier_name")
        assert False, "Failed to raise ValueError on invalid classifier name"
    except ValueError as e:
        print(f"  [PASS] get_classifier('invalid_classifier_name') correctly raised ValueError: {e}")

    # Build Pipeline A with custom parameters
    pipe_a = create_pipeline_a(classifier=clf_custom, vectorizer_params=custom_vec_params)
    tfidf_a = pipe_a.named_steps["tfidf"]
    assert tfidf_a.min_df == 5, f"Expected min_df=5, got {tfidf_a.min_df}"
    assert tfidf_a.ngram_range == (1, 3), f"Expected ngram_range=(1, 3), got {tfidf_a.ngram_range}"
    assert tfidf_a.sublinear_tf is False, f"Expected sublinear_tf=False, got {tfidf_a.sublinear_tf}"
    assert tfidf_a.max_features == 1000, f"Expected max_features=1000, got {tfidf_a.max_features}"
    assert pipe_a.named_steps["classifier"].C == 0.5
    print("  [PASS] Pipeline A custom parameters propagated correctly to TfidfVectorizer and Classifier.")

    # Build Pipeline C with custom parameters
    pipe_c = create_pipeline_c(classifier=clf_custom, vectorizer_params=custom_vec_params)
    union_c = pipe_c.named_steps["features"]
    tfidf_c = union_c.transformer_list[0][1].named_steps["tfidf"]
    assert tfidf_c.min_df == 5, f"Expected min_df=5, got {tfidf_c.min_df}"
    assert tfidf_c.ngram_range == (1, 3), f"Expected ngram_range=(1, 3), got {tfidf_c.ngram_range}"
    assert tfidf_c.sublinear_tf is False, f"Expected sublinear_tf=False, got {tfidf_c.sublinear_tf}"
    assert tfidf_c.max_features == 1000, f"Expected max_features=1000, got {tfidf_c.max_features}"
    print("  [PASS] Pipeline C custom parameters propagated correctly into FeatureUnion tfidf_branch.")


def test_3_sparse_matrix_format_nonnegativity_and_features():
    print("\n" + "=" * 70)
    print("TEST 3: SPARSE FORMAT, NON-NEGATIVITY & FEATURE EXTRACTION")
    print("=" * 70)

    emails, labels = generate_synthetic_emails()
    print(f"  Generated {len(emails)} synthetic emails (10 Ham, 10 Spam).")

    # 1. Test un-fitted exception on extract_feature_names()
    pipe_a_unfitted = create_pipeline_a()
    pipe_b_unfitted = create_pipeline_b()
    pipe_c_unfitted = create_pipeline_c()

    for p, label in [(pipe_a_unfitted, "Pipeline A"), (pipe_b_unfitted, "Pipeline B"), (pipe_c_unfitted, "Pipeline C")]:
        try:
            extract_feature_names(p)
            assert False, f"{label} did not raise ValueError when un-fitted"
        except ValueError as e:
            assert "must be fitted" in str(e).lower(), f"Unexpected error message: {e}"
            print(f"  [PASS] {label} correctly raised ValueError when un-fitted: '{e}'")

    # 2. Fit transform pipelines
    X_a = pipe_a_unfitted.fit_transform(emails)
    X_b = pipe_b_unfitted.fit_transform(emails)
    X_c = pipe_c_unfitted.fit_transform(emails)

    for X, name in [(X_a, "Pipeline A"), (X_b, "Pipeline B"), (X_c, "Pipeline C")]:
        assert issparse(X), f"{name} output is not sparse"
        assert isinstance(X, csr_matrix), f"{name} output is not strictly csr_matrix, got {type(X)}"
        print(f"  [PASS] {name} output format verified: issparse=True, type={type(X).__name__}, shape={X.shape}")

    # 3. Assert non-negativity for Pipeline C
    assert np.all(X_c.data >= 0.0), "Pipeline C combined matrix contains negative values!"
    min_c = float(np.min(X_c.data)) if X_c.nnz > 0 else 0.0
    print(f"  [PASS] Pipeline C strictly non-negative: min non-zero value = {min_c:.6f} >= 0.0")

    # 4. Extract feature names and assert alignment
    names_a = extract_feature_names(pipe_a_unfitted)
    names_b = extract_feature_names(pipe_b_unfitted)
    names_c = extract_feature_names(pipe_c_unfitted)

    assert names_a.shape[0] == X_a.shape[1], f"Pipeline A dimension mismatch: {names_a.shape[0]} vs {X_a.shape[1]}"
    assert names_b.shape[0] == X_b.shape[1], f"Pipeline B dimension mismatch: {names_b.shape[0]} vs {X_b.shape[1]}"
    assert names_c.shape[0] == X_c.shape[1], f"Pipeline C dimension mismatch: {names_c.shape[0]} vs {X_c.shape[1]}"
    print(f"  [PASS] Feature count alignment confirmed:")
    print(f"         Pipeline A: {names_a.shape[0]} features == {X_a.shape[1]} columns")
    print(f"         Pipeline B: {names_b.shape[0]} features == {X_b.shape[1]} columns")
    print(f"         Pipeline C: {names_c.shape[0]} features == {X_c.shape[1]} columns")

    # 5. Check Pipeline C heuristic prefix
    heur_prefixed = [fn for fn in names_c if fn.startswith("heuristic_branch__")]
    expected_heur_names = [
        "heuristic_branch__uppercase_ratio",
        "heuristic_branch__all_caps_word_count",
        "heuristic_branch__exclamation_count",
        "heuristic_branch__question_count",
        "heuristic_branch__currency_symbol_count",
        "heuristic_branch__url_count",
    ]
    assert len(heur_prefixed) == 6, f"Expected 6 heuristic features, found {len(heur_prefixed)}"
    assert heur_prefixed == expected_heur_names, f"Heuristic names do not match expected: {heur_prefixed}"
    print(f"  [PASS] All 6 heuristic features confirmed with 'heuristic_branch__' namespace prefix:")
    for h in heur_prefixed:
        print(f"         - {h}")


def test_4_six_classifier_fit_and_predict():
    print("\n" + "=" * 70)
    print("TEST 4: SIX-CLASSIFIER FIT & PREDICT WITH PIPELINE C")
    print("=" * 70)

    emails, labels = generate_synthetic_emails()

    classifier_keys = [
        "logistic_regression",
        "multinomial_nb",
        "decision_tree",
        "random_forest",
        "linear_svc",
        "knn",
    ]

    for key in classifier_keys:
        clf = get_classifier(key)
        pipeline = create_pipeline_c(classifier=clf)

        # Fit
        pipeline.fit(emails, labels)

        # Predict
        preds = pipeline.predict(emails)

        # Assertions
        assert isinstance(preds, np.ndarray), f"{key} predictions not a numpy ndarray"
        assert preds.shape == (20,), f"{key} predictions shape mismatch: {preds.shape} vs (20,)"
        assert np.issubdtype(preds.dtype, np.integer), f"{key} predictions dtype is not integer: {preds.dtype}"
        unique_preds = np.unique(preds)
        print(f"  [PASS] {key:<22}: fit() & predict() successful. Shape={preds.shape}, dtype={preds.dtype}, classes={unique_preds.tolist()}")


def test_5_workspace_integrity():
    print("\n" + "=" * 70)
    print("TEST 5: WORKSPACE & DATASET INTEGRITY VERIFICATION")
    print("=" * 70)

    def calc_md5(filepath: Path) -> str:
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    # 1. Dataset checksums
    raw_path = workspace_dir / "dataset" / "spam_assassin.csv"
    clean_path = workspace_dir / "dataset" / "spam_assassin_cleaned.csv"

    expected_raw_md5 = "cc161a4efd9534c8a32664a52942d097"
    expected_clean_md5 = "87ab383a0a2853ff148a0e61fb5f2e5d"

    actual_raw_md5 = calc_md5(raw_path)
    actual_clean_md5 = calc_md5(clean_path)

    assert actual_raw_md5 == expected_raw_md5, f"RAW MD5 mismatch: {actual_raw_md5} != {expected_raw_md5}"
    assert actual_clean_md5 == expected_clean_md5, f"CLEAN MD5 mismatch: {actual_clean_md5} != {expected_clean_md5}"
    print(f"  [PASS] dataset/spam_assassin.csv MD5         : {actual_raw_md5} (Matches expected)")
    print(f"  [PASS] dataset/spam_assassin_cleaned.csv MD5 : {actual_clean_md5} (Matches expected)")

    # 2. src/preprocessing.py line count
    prep_path = workspace_dir / "src" / "preprocessing.py"
    with open(prep_path, "r", encoding="utf-8") as f:
        prep_lines = len(f.readlines())
    assert prep_lines == 487, f"src/preprocessing.py line count changed: {prep_lines} != 487"
    print(f"  [PASS] src/preprocessing.py line count       : {prep_lines} (Strictly 487 lines)")

    # 3. models/ cleanliness
    models_dir = workspace_dir / "models"
    model_files = [f.name for f in models_dir.glob("*")]
    assert model_files == [".gitkeep"], f"models/ contains unexpected files: {model_files}"
    print(f"  [PASS] models/ directory cleanliness         : {model_files} (Only .gitkeep present)")


def main():
    print("=" * 70)
    print("STARTING STAGE 5B PIPELINE FACTORY INTEGRITY TEST SUITE")
    print("=" * 70)

    test_1_factory_import_and_cloning()
    test_2_parameter_propagation()
    test_3_sparse_matrix_format_nonnegativity_and_features()
    test_4_six_classifier_fit_and_predict()
    test_5_workspace_integrity()

    print("\n" + "=" * 70)
    print("ALL STAGE 5B UNIT AND INTEGRATION TESTS PASSED WITH ZERO ERRORS.")
    print("=" * 70)


if __name__ == "__main__":
    main()

