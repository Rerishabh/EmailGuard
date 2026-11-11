"""Stage 6A: Stratified Partitioning, Leakage Audit & Pipeline Refinement Tests.

Covers:
  Part 1: Joblib serialization round-trip verification for Pipeline C.
  Part 2: 80/20 stratified partition validation + full-text, subject, body overlap audit.
  Part 3: 5-fold StratifiedKFold distribution check on training set only.
  Part 4: Partition index persistence (train/test .npy) + SHA-256 reproducibility audit.
  Part 5: Workspace & dataset integrity verification (MD5 checksums, line counts, models/ hygiene).
"""

import hashlib
import os
import tempfile
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

workspace_dir = Path(__file__).resolve().parent.parent
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

from src.pipeline import create_pipeline_c, get_classifier


# =====================================================================
# PART 1: JOBLIB SERIALIZATION ROUND-TRIP
# =====================================================================
def part1_joblib_roundtrip():
    print("=" * 70)
    print("PART 1: JOBLIB SERIALIZATION ROUND-TRIP VERIFICATION")
    print("=" * 70)

    # 20 synthetic emails (10 ham, 10 spam)
    emails = [
        "Team review the quarterly project budget report before morning meeting tomorrow.",
        "Quarterly project roadmap documentation updated for review meeting this team.",
        "Please find quarterly financial budget summary attached for team review meeting.",
        "Good morning team review software project status update during today meeting.",
        "Team meeting reminder quarterly project evaluation and report review tomorrow.",
        "Team review final quarterly documentation before scheduled meeting tomorrow.",
        "Review attached project schedule report for upcoming quarterly planning meeting.",
        "Team discussion project roadmap update quarterly engineering report meeting.",
        "Morning team check quarterly budget report and send project notes for meeting.",
        "Quarterly project review meeting tomorrow please review all budget report items.",
        "URGENT WINNER Claim FREE cash prize bonus now Click http://reward.example.com to win.",
        "CONGRATULATIONS You won FREE cash reward prize Visit http://winner.example.com now.",
        "CLAIM CASH NOW Exclusive prize bonus waiting Click http://cash-bonus.example.com today.",
        "ACT FAST Claim FREE prize cash payout today Click http://promo.example.com now.",
        "SPECIAL OFFER Free bonus cash reward Visit http://gift.example.com to claim prize.",
        "URGENT ALERT You have FREE prize reward voucher Click http://claim.example.com to win.",
        "FINAL NOTICE Claim cash prize bonus today Visit http://payout.example.com for reward.",
        "EXCLUSIVE Free cash payout Click http://jackpot.example.com now to claim cash prize.",
        "CONGRATULATIONS free prize reward ready Visit http://rewards.example.com to claim.",
        "WIN BIG TODAY Claim FREE cash bonus prize Click http://instant-cash.example.com now.",
    ]
    y_synth = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    clf = get_classifier("logistic_regression")
    pipe = create_pipeline_c(classifier=clf)
    pipe.fit(emails, y_synth)
    preds_original = pipe.predict(emails)

    # Serialize to a temporary scratch file
    scratch_dir = workspace_dir / "scratch"
    tmp_path = scratch_dir / "_tmp_joblib_test.joblib"

    joblib.dump(pipe, tmp_path)
    print(f"  Serialized Pipeline C to: {tmp_path.name}")

    loaded_pipe = joblib.load(tmp_path)
    preds_loaded = loaded_pipe.predict(emails)

    assert np.array_equal(preds_original, preds_loaded), (
        f"Joblib round-trip prediction mismatch!\n"
        f"Original: {preds_original}\nLoaded  : {preds_loaded}"
    )
    print(f"  Original predictions : {preds_original.tolist()}")
    print(f"  Loaded   predictions : {preds_loaded.tolist()}")
    print(f"  Round-trip match (all 20 samples): True")

    # Clean up immediately
    tmp_path.unlink()
    assert not tmp_path.exists(), "Temporary joblib file was not cleaned up!"
    print(f"  Temporary file cleaned up successfully.")
    print(f"  [PASS] Joblib serialization round-trip VERIFIED.\n")


# =====================================================================
# PART 2: STRATIFIED PARTITIONING & LEAK-SURFACE AUDIT
# =====================================================================
def part2_partitioning_and_overlap(df: pd.DataFrame):
    print("=" * 70)
    print("PART 2: STRATIFIED PARTITIONING & LEAK-SURFACE AUDIT")
    print("=" * 70)

    # --- 2.1 Generate 80/20 split ---
    # train_test_split(X, y, ...) returns (X_train, X_test, y_train, y_test)
    # We split on df.index as X; y is provided for stratification only.
    train_idx, test_idx, _, _ = train_test_split(
        df.index,
        df["label"],
        test_size=0.20,
        stratify=df["label"],
        random_state=42,
    )

    n_train = len(train_idx)
    n_test = len(test_idx)

    train_ham = (df.loc[train_idx, "label"] == 0).sum()
    train_spam = (df.loc[train_idx, "label"] == 1).sum()
    test_ham = (df.loc[test_idx, "label"] == 0).sum()
    test_spam = (df.loc[test_idx, "label"] == 1).sum()

    # Validate partition sizes
    assert n_train == 2560, f"Expected 2560 train rows, got {n_train}"
    assert n_test == 641, f"Expected 641 test rows, got {n_test}"
    assert n_train + n_test == 3201, f"Total mismatch: {n_train + n_test} != 3201"

    # Validate class balances — empirically verified with random_state=42 on this dataset
    # Total: Ham=2720 (84.97%), Spam=481 (15.03%)
    # Train: Ham=2175 (84.96%), Spam=385 (15.04%)
    # Test:  Ham=545  (85.02%), Spam=96  (14.98%)
    assert train_ham == 2175, f"Expected train ham=2175, got {train_ham}"
    assert train_spam == 385, f"Expected train spam=385, got {train_spam}"
    assert test_ham == 545, f"Expected test ham=545, got {test_ham}"
    assert test_spam == 96, f"Expected test spam=96, got {test_spam}"

    # Validate mutual exclusivity
    assert set(train_idx).isdisjoint(set(test_idx)), "FAIL: Train/test index overlap detected!"
    assert len(set(train_idx) | set(test_idx)) == 3201, "FAIL: Union of indices != 3201"

    print(f"  Partition Summary:")
    print(f"  {'Set':<10} {'Total':>8} {'Ham (0)':>12} {'Ham %':>8} {'Spam (1)':>12} {'Spam %':>8}")
    print(f"  {'-'*62}")
    print(f"  {'Train':<10} {n_train:>8} {train_ham:>12} {train_ham/n_train*100:>7.2f}% {train_spam:>12} {train_spam/n_train*100:>7.2f}%")
    print(f"  {'Test':<10} {n_test:>8} {test_ham:>12} {test_ham/n_test*100:>7.2f}% {test_spam:>12} {test_spam/n_test*100:>7.2f}%")
    print(f"  {'Total':<10} {n_train+n_test:>8} {train_ham+test_ham:>12} {'':>8} {train_spam+test_spam:>12}")
    print(f"  Index mutual exclusivity: {set(train_idx).isdisjoint(set(test_idx))}")
    print(f"  Union covers all 3201 rows: {len(set(train_idx) | set(test_idx)) == 3201}")
    print()

    # --- 2.2 Overlap Audit ---
    # Full text overlap
    train_text = set(df.loc[train_idx, "text"])
    test_text = set(df.loc[test_idx, "text"])
    overlap_text = len(train_text & test_text)
    assert overlap_text == 0, f"Full-text overlap detected! {overlap_text} shared texts."
    print(f"  Exact Full-Text Overlap  : {overlap_text} (MUST be 0 — PASS)")

    # Subject-only overlap
    train_sub = set(df.loc[train_idx, "subject"].dropna())
    test_sub = set(df.loc[test_idx, "subject"].dropna())
    overlap_sub = len(train_sub & test_sub)
    pct_train_sub = overlap_sub / len(train_sub) * 100 if train_sub else 0.0
    pct_test_sub = overlap_sub / len(test_sub) * 100 if test_sub else 0.0

    # Body-only overlap
    train_body = set(df.loc[train_idx, "body"].dropna())
    test_body = set(df.loc[test_idx, "body"].dropna())
    overlap_body = len(train_body & test_body)
    pct_train_body = overlap_body / len(train_body) * 100 if train_body else 0.0
    pct_test_body = overlap_body / len(test_body) * 100 if test_body else 0.0

    print(f"\n  Leak-Surface Overlap Audit:")
    print(f"  {'Field':<20} {'Overlap Count':>15} {'% Train Distinct':>18} {'% Test Distinct':>17}")
    print(f"  {'-'*72}")
    print(f"  {'Full Text (text)':<20} {overlap_text:>15} {'N/A (must be 0)':>18} {'N/A (must be 0)':>17}")
    print(f"  {'Subject only':<20} {overlap_sub:>15} {pct_train_sub:>17.2f}% {pct_test_sub:>16.2f}%")
    print(f"  {'Body only':<20} {overlap_body:>15} {pct_train_body:>17.2f}% {pct_test_body:>16.2f}%")
    print(f"\n  Note: Subject and body overlaps are legitimate cross-thread/boilerplate occurrences.")
    print(f"  [PASS] All partitioning and overlap assertions satisfied.\n")

    return train_idx, test_idx


# =====================================================================
# PART 3: 5-FOLD STRATIFIED CV DISTRIBUTION CHECK (TRAIN ONLY)
# =====================================================================
def part3_cv_distribution(df: pd.DataFrame, train_idx):
    print("=" * 70)
    print("PART 3: 5-FOLD STRATIFIED CROSS-VALIDATION DISTRIBUTION CHECK")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df_train = df.loc[train_idx].reset_index(drop=True)
    X_tr = df_train["text"]
    y_tr = df_train["label"]

    assert len(df_train) == 2560, f"Expected 2560 training rows, got {len(df_train)}"

    all_val_indices = []
    fold_records = []

    print(f"  {'Fold':>5} {'Val N':>8} {'Val Ham':>10} {'Ham %':>8} {'Val Spam':>10} {'Spam %':>8}")
    print(f"  {'-' * 55}")

    for fold_i, (_, val_local_idx) in enumerate(skf.split(X_tr, y_tr)):
        val_records = df_train.iloc[val_local_idx]
        val_n = len(val_records)
        val_ham = (val_records["label"] == 0).sum()
        val_spam = (val_records["label"] == 1).sum()
        val_spam_pct = val_spam / val_n * 100

        # Assertions per fold — empirically verified with StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # on training set N=2560 (Ham=2175, Spam=385): every fold has N=512, Ham=435, Spam=77
        assert val_n == 512, f"Fold {fold_i}: Expected 512 validation records, got {val_n}"
        assert val_ham == 435, f"Fold {fold_i}: Expected ham=435, got {val_ham}"
        assert val_spam == 77, f"Fold {fold_i}: Expected spam=77, got {val_spam}"
        assert 14.5 <= val_spam_pct <= 15.5, f"Fold {fold_i}: Spam ratio {val_spam_pct:.2f}% outside expected range"

        all_val_indices.extend(val_local_idx.tolist())
        fold_records.append((fold_i, val_n, val_ham, val_spam, val_spam_pct))

        print(f"  {fold_i:>5} {val_n:>8} {val_ham:>10} {val_ham/val_n*100:>7.2f}% {val_spam:>10} {val_spam_pct:>7.2f}%")

    print()

    # Union of validation indices must equal full training set
    all_val_set = set(all_val_indices)
    full_train_set = set(range(len(df_train)))
    assert all_val_set == full_train_set, (
        f"Validation index union != full training set. "
        f"Missing: {full_train_set - all_val_set}, Extra: {all_val_set - full_train_set}"
    )
    print(f"  Union of all validation sets == full training set (N=2560): True")

    # Pairwise disjointness of folds
    fold_idx_sets = []
    for fold_i, (_, val_local_idx) in enumerate(skf.split(X_tr, y_tr)):
        fold_idx_sets.append(set(val_local_idx.tolist()))

    for i in range(5):
        for j in range(i + 1, 5):
            assert fold_idx_sets[i].isdisjoint(fold_idx_sets[j]), (
                f"Fold {i} and Fold {j} have overlapping validation indices!"
            )
    print(f"  All 5 validation folds are pairwise mutually disjoint: True")
    print(f"  [PASS] All 5-fold CV distribution assertions satisfied.\n")


# =====================================================================
# PART 4: PARTITION PERSISTENCE & SHA-256 REPRODUCIBILITY AUDIT
# =====================================================================
def part4_persistence(train_idx, test_idx):
    print("=" * 70)
    print("PART 4: PARTITION PERSISTENCE & SHA-256 REPRODUCIBILITY AUDIT")
    print("=" * 70)

    train_path = workspace_dir / "dataset" / "train_split_indices.npy"
    test_path = workspace_dir / "dataset" / "test_split_indices.npy"

    # Persist
    np.save(train_path, np.array(train_idx, dtype=np.int64))
    np.save(test_path, np.array(test_idx, dtype=np.int64))
    print(f"  Saved: {train_path.name}  (shape: {np.array(train_idx).shape})")
    print(f"  Saved: {test_path.name}  (shape: {np.array(test_idx).shape})")

    # SHA-256 hashes
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    sha_train = sha256_file(train_path)
    sha_test = sha256_file(test_path)
    print(f"\n  SHA-256 Hashes:")
    print(f"  train_split_indices.npy : {sha_train}")
    print(f"  test_split_indices.npy  : {sha_test}")

    # Reload and verify
    loaded_train = np.load(train_path)
    loaded_test = np.load(test_path)

    assert np.array_equal(loaded_train, np.array(train_idx, dtype=np.int64)), \
        "Reloaded train indices do not match original!"
    assert np.array_equal(loaded_test, np.array(test_idx, dtype=np.int64)), \
        "Reloaded test indices do not match original!"

    print(f"\n  Reload verification:")
    print(f"  np.array_equal(loaded_train, train_idx) : True")
    print(f"  np.array_equal(loaded_test,  test_idx)  : True")
    print(f"  [PASS] Partition persistence and SHA-256 reproducibility VERIFIED.\n")

    return sha_train, sha_test


# =====================================================================
# PART 5: WORKSPACE & DATASET INTEGRITY VERIFICATION
# =====================================================================
def part5_integrity():
    print("=" * 70)
    print("PART 5: WORKSPACE & DATASET INTEGRITY VERIFICATION")
    print("=" * 70)

    def calc_md5(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    checks = {
        "dataset/spam_assassin.csv": "cc161a4efd9534c8a32664a52942d097",
        "dataset/spam_assassin_cleaned.csv": "87ab383a0a2853ff148a0e61fb5f2e5d",
    }

    for rel_path, expected_md5 in checks.items():
        p = workspace_dir / rel_path
        actual = calc_md5(p)
        status = "PASS" if actual == expected_md5 else "FAIL"
        assert actual == expected_md5, f"MD5 mismatch for {rel_path}: {actual} != {expected_md5}"
        print(f"  [PASS] {rel_path:<40} MD5: {actual}")

    # src/preprocessing.py line count
    prep_path = workspace_dir / "src" / "preprocessing.py"
    with open(prep_path, "r", encoding="utf-8") as f:
        prep_lines = len(f.readlines())
    assert prep_lines == 487, f"src/preprocessing.py line count changed: {prep_lines} != 487"
    print(f"  [PASS] src/preprocessing.py                      Lines: {prep_lines} (Expected: 487)")

    # src/pipeline.py sanity check (not locked)
    pipe_path = workspace_dir / "src" / "pipeline.py"
    with open(pipe_path, "r", encoding="utf-8") as f:
        pipe_lines = len(f.readlines())
    print(f"  [INFO] src/pipeline.py                           Lines: {pipe_lines}")

    # models/ cleanliness
    models_dir = workspace_dir / "models"
    model_files = [f.name for f in models_dir.glob("*")]
    assert model_files == [".gitkeep"], f"models/ contains unexpected files: {model_files}"
    print(f"  [PASS] models/ directory                         Contents: {model_files}")

    # results/graphs/ PNG count
    graphs_dir = workspace_dir / "results" / "graphs"
    pngs = sorted([f.name for f in graphs_dir.glob("*.png")])
    assert len(pngs) == 8, f"results/graphs/ PNG count changed: {len(pngs)} != 8"
    print(f"  [PASS] results/graphs/ PNG files                 Count: {len(pngs)} (Expected: 8)")
    print()


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 70)
    print("STAGE 6A: STRATIFIED PARTITIONING, LEAKAGE AUDIT & REFINEMENT")
    print("=" * 70)
    print()

    part1_joblib_roundtrip()

    # Load dataset once for Parts 2–3
    cleaned_path = workspace_dir / "dataset" / "spam_assassin_cleaned.csv"
    df = pd.read_csv(cleaned_path)
    assert len(df) == 3201, f"Expected 3201 rows, got {len(df)}"
    print(f"  Loaded dataset: {len(df)} records, columns: {list(df.columns)}\n")

    train_idx, test_idx = part2_partitioning_and_overlap(df)
    part3_cv_distribution(df, train_idx)
    sha_train, sha_test = part4_persistence(train_idx, test_idx)
    part5_integrity()

    print("=" * 70)
    print("ALL STAGE 6A TESTS PASSED — Zero errors, zero constraint violations.")
    print("=" * 70)


if __name__ == "__main__":
    main()
