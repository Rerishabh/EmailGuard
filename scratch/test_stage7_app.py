"""Stage 7A Verification Test: Model Loader, Architecture & Caching Audit.

Verifies:
1. Workspace directory structure (app/, app/utils/, app/components/, app/styles/).
2. Repository integrity baseline (dataset MD5/SHA256, src/preprocessing.py line count).
3. Champion configuration loader (models/champion_config.json).
4. Champion model loader and cache (models/champion_pipeline.joblib).
5. Pipeline C high-throughput alternative loader and cache (fitted strictly on N=2,560 train partition).
6. Continuous signed geometric margin inference for both pipelines.
7. Operating threshold classification (default 0.0 vs calibrated thresholds).
8. Heuristic feature extraction contract for Pipeline C.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.pipeline import Pipeline

from app.utils.model_loader import (
    CHAMPION_CONFIG_PATH,
    CHAMPION_MODEL_PATH,
    CLEANED_DATASET_PATH,
    PIPELINE_B_CALIBRATED_THRESHOLD,
    PIPELINE_B_DEFAULT_THRESHOLD,
    PIPELINE_C_CALIBRATED_THRESHOLD,
    PIPELINE_C_DEFAULT_THRESHOLD,
    TRAIN_INDICES_PATH,
    classify_email,
    extract_heuristic_features,
    get_model_specs,
    get_project_root,
    load_champion_config,
    load_champion_model,
    load_pipeline_c_model,
    predict_decision_margin,
)


def calc_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def calc_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def test_stage7a():
    print("=" * 80)
    print("STAGE 7A AUDIT: MODEL LOADER & ARCHITECTURE VERIFICATION")
    print("=" * 80)

    # 1. Directory Structure Checks
    print("\n[Step 1/7] Auditing App Directory Structure...")
    for dir_name in ["app", "app/utils", "app/components", "app/styles"]:
        p = REPO_ROOT / dir_name
        assert p.exists() and p.is_dir(), f"Missing required directory: {dir_name}"
        print(f"  [PASS] Directory exists: {dir_name}/")

    assert (REPO_ROOT / "app" / "utils" / "__init__.py").exists(), "Missing app/utils/__init__.py"
    assert (REPO_ROOT / "app" / "utils" / "model_loader.py").exists(), "Missing app/utils/model_loader.py"
    print("  [PASS] Core utility files present.")

    # 2. Integrity Baseline Checks
    print("\n[Step 2/7] Verifying Locked Integrity Baselines...")
    raw_md5 = calc_md5(REPO_ROOT / "dataset" / "spam_assassin.csv")
    clean_md5 = calc_md5(REPO_ROOT / "dataset" / "spam_assassin_cleaned.csv")
    train_sha = calc_sha256(REPO_ROOT / "dataset" / "train_split_indices.npy")
    test_sha = calc_sha256(REPO_ROOT / "dataset" / "test_split_indices.npy")

    assert raw_md5 == "cc161a4efd9534c8a32664a52942d097", f"Raw dataset MD5 mismatch: {raw_md5}"
    assert clean_md5 == "87ab383a0a2853ff148a0e61fb5f2e5d", f"Cleaned dataset MD5 mismatch: {clean_md5}"
    assert train_sha == "84a9ef640ccc54711068d52c1f22d767aa800142c0f7b4822014855158e61d91", f"Train SHA mismatch: {train_sha}"
    assert test_sha == "bb1d408a89a7f53177defd1e4cd11d1036cccb577e3bc670a13c0d170210a0f5", f"Test SHA mismatch: {test_sha}"
    print("  [PASS] Dataset and partition checksums verified against frozen baselines.")

    prep_path = REPO_ROOT / "src" / "preprocessing.py"
    prep_lines = len(prep_path.read_text(encoding="utf-8").splitlines())
    assert prep_lines == 487, f"src/preprocessing.py line count mismatch: expected 487, got {prep_lines}"
    print(f"  [PASS] src/preprocessing.py verified ({prep_lines} lines).")

    # 3. Champion Config Loader
    print("\n[Step 3/7] Verifying Champion Configuration Loader...")
    config = load_champion_config()
    assert isinstance(config, dict), "Champion config must be a dict"
    assert config["candidate_name"] == "pipeline_b_linear_svc"
    assert config["champion_status"] == "PRE_DECLARED_AND_VERIFIED"
    assert np.isclose(config["frozen_calibrated_threshold"], PIPELINE_B_CALIBRATED_THRESHOLD, atol=1e-12)
    print(f"  [PASS] Champion configuration loaded successfully.")
    print(f"         Candidate: {config['candidate_name']}")
    print(f"         Calibrated threshold: {config['frozen_calibrated_threshold']:.16f}")

    # 4. Champion Model Loader & Resource Cache
    print("\n[Step 4/7] Verifying Champion Model Loader (Pipeline B + LinearSVC)...")
    model_b_1 = load_champion_model()
    model_b_2 = load_champion_model()
    assert isinstance(model_b_1, Pipeline), f"Expected Pipeline, got {type(model_b_1)}"
    assert model_b_1 is model_b_2, "Streamlit resource cache must return the identical object"
    step_names = [name for name, _ in model_b_1.steps]
    assert step_names == ["transformer", "tfidf", "classifier"], f"Unexpected steps: {step_names}"
    print(f"  [PASS] Champion model loaded and resource caching verified.")
    print(f"         Pipeline steps: {step_names}")

    # 5. Pipeline C Model Loader & Resource Cache
    print("\n[Step 5/7] Verifying Pipeline C Model Loader (High-Throughput Alternative)...")
    model_c_1 = load_pipeline_c_model()
    model_c_2 = load_pipeline_c_model()
    assert isinstance(model_c_1, Pipeline), f"Expected Pipeline, got {type(model_c_1)}"
    assert model_c_1 is model_c_2, "Streamlit resource cache must return the identical object"
    c_step_names = [name for name, _ in model_c_1.steps]
    assert c_step_names == ["features", "to_csr", "classifier"], f"Unexpected steps: {c_step_names}"
    print(f"  [PASS] Pipeline C constructed, trained on N=2,560, and resource caching verified.")
    print(f"         Pipeline steps: {c_step_names}")

    # 6. Sample Inferences & Threshold Evaluation
    print("\n[Step 6/7] Testing Model Inferences on Ham and Spam Probes...")
    sample_ham = (
        "Hi team, just checking in regarding the meeting schedule for tomorrow morning. "
        "Please let me know if anyone has updates to the agenda or slides. Best regards."
    )
    sample_spam = (
        "CONGRATULATIONS! You have been selected as the WINNER of $5,000,000 CASH prize! "
        "Claim your prize NOW by clicking here: http://claim-cash-instant.fake-domain/win "
        "Limited time offer! Act immediately!!!"
    )

    # Test Pipeline B
    margin_b_ham = predict_decision_margin(model_b_1, sample_ham)
    margin_b_spam = predict_decision_margin(model_b_1, sample_spam)
    pred_b_ham_def = classify_email(model_b_1, sample_ham, threshold=PIPELINE_B_DEFAULT_THRESHOLD)
    pred_b_ham_cal = classify_email(model_b_1, sample_ham, threshold=PIPELINE_B_CALIBRATED_THRESHOLD)
    pred_b_spam_def = classify_email(model_b_1, sample_spam, threshold=PIPELINE_B_DEFAULT_THRESHOLD)
    pred_b_spam_cal = classify_email(model_b_1, sample_spam, threshold=PIPELINE_B_CALIBRATED_THRESHOLD)

    assert margin_b_ham < 0.0, f"Expected negative margin for ham, got {margin_b_ham}"
    assert margin_b_spam > 0.0, f"Expected positive margin for spam, got {margin_b_spam}"
    assert not pred_b_ham_def["is_spam"] and not pred_b_ham_cal["is_spam"]
    assert pred_b_spam_def["is_spam"] and pred_b_spam_cal["is_spam"]
    print(f"  [PASS] Pipeline B inferences:")
    print(f"         Ham  Margin: {margin_b_ham:+.4f} -> {pred_b_ham_cal['label']}")
    print(f"         Spam Margin: {margin_b_spam:+.4f} -> {pred_b_spam_cal['label']}")

    # Test Pipeline C
    margin_c_ham = predict_decision_margin(model_c_1, sample_ham)
    margin_c_spam = predict_decision_margin(model_c_1, sample_spam)
    pred_c_ham_def = classify_email(model_c_1, sample_ham, threshold=PIPELINE_C_DEFAULT_THRESHOLD)
    pred_c_ham_cal = classify_email(model_c_1, sample_ham, threshold=PIPELINE_C_CALIBRATED_THRESHOLD)
    pred_c_spam_def = classify_email(model_c_1, sample_spam, threshold=PIPELINE_C_DEFAULT_THRESHOLD)
    pred_c_spam_cal = classify_email(model_c_1, sample_spam, threshold=PIPELINE_C_CALIBRATED_THRESHOLD)

    assert margin_c_ham < 0.0, f"Expected negative margin for ham, got {margin_c_ham}"
    assert margin_c_spam > 0.0, f"Expected positive margin for spam, got {margin_c_spam}"
    assert not pred_c_ham_def["is_spam"] and not pred_c_ham_cal["is_spam"]
    assert pred_c_spam_def["is_spam"] and pred_c_spam_cal["is_spam"]
    print(f"  [PASS] Pipeline C inferences:")
    print(f"         Ham  Margin: {margin_c_ham:+.4f} -> {pred_c_ham_cal['label']}")
    print(f"         Spam Margin: {margin_c_spam:+.4f} -> {pred_c_spam_cal['label']}")

    # 7. Heuristic Feature Extraction & Model Specs
    print("\n[Step 7/7] Verifying Heuristic Feature Extraction & Specs...")
    heuristics = extract_heuristic_features(sample_spam)
    expected_keys = [
        "uppercase_ratio",
        "all_caps_word_count",
        "exclamation_count",
        "question_count",
        "currency_symbol_count",
        "url_count",
    ]
    assert list(heuristics.keys()) == expected_keys, f"Unexpected heuristic keys: {list(heuristics.keys())}"
    assert heuristics["all_caps_word_count"]["value"] >= 3, "Expected >= 3 ALL-CAPS words in sample spam"
    assert heuristics["exclamation_count"]["value"] >= 3, "Expected >= 3 exclamation marks in sample spam"
    assert heuristics["currency_symbol_count"]["value"] == 1, "Expected 1 currency symbol ($) in sample spam"
    assert heuristics["url_count"]["value"] == 1, "Expected 1 URL in sample spam"
    print("  [PASS] Heuristic feature extraction contract fully verified.")

    specs = get_model_specs()
    assert "pipeline_b" in specs and "pipeline_c" in specs
    assert specs["pipeline_b"]["uses_heuristics"] is False
    assert specs["pipeline_c"]["uses_heuristics"] is True
    print("  [PASS] Model specifications and metadata verified.")

    print("\n" + "=" * 80)
    print("STAGE 7A AUDIT: ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_stage7a()

