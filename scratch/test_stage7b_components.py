"""Stage 7B Verification Harness: UI Components & Enterprise Styling Verification.

Audits and verifies:
1. File structure of all Stage 7B deliverables.
2. Clean imports of app.components submodules.
3. Rigorous preset validation across both models (Pipeline B & Pipeline C) using calibrated thresholds.
4. Visualization component validation across diverse margin edge cases.
5. Prediction component validation with security verdicts, metrics, and heuristics.
6. Dataset and source code integrity baselines (487 lines, MD5 & SHA-256 hashes).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.components.email_examples import get_email_examples
from app.components.prediction import generate_prediction_card_html, render_prediction_card
from app.components.visualization import generate_margin_gauge_svg, render_margin_gauge
from app.utils.model_loader import (
    PIPELINE_B_CALIBRATED_THRESHOLD,
    PIPELINE_C_CALIBRATED_THRESHOLD,
    classify_email,
    extract_heuristic_features,
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


def test_stage7b():
    print("=" * 90)
    print("STAGE 7B AUDIT: UI COMPONENTS & ENTERPRISE DESIGN VERIFICATION")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # TEST 1: FILE STRUCTURE
    # -------------------------------------------------------------------------
    print("\n[Test 1/6] Verifying Stage 7B File Deliverables...")
    required_files = [
        REPO_ROOT / "app" / "components" / "__init__.py",
        REPO_ROOT / "app" / "components" / "email_examples.py",
        REPO_ROOT / "app" / "components" / "prediction.py",
        REPO_ROOT / "app" / "components" / "visualization.py",
        REPO_ROOT / "app" / "styles" / "custom.css",
    ]
    for rf in required_files:
        assert rf.exists() and rf.is_file(), f"Missing required file: {rf}"
        print(f"  [PASS] Found: {rf.relative_to(REPO_ROOT)}")

    css_content = (REPO_ROOT / "app" / "styles" / "custom.css").read_text(encoding="utf-8")
    assert "--ham-color: #16A34A" in css_content, "Missing --ham-color design token"
    assert "--spam-color: #DC2626" in css_content, "Missing --spam-color design token"
    assert "--bg-main: #F8FAFC" in css_content, "Missing --bg-main design token"
    print("  [PASS] custom.css design tokens verified.")

    # -------------------------------------------------------------------------
    # TEST 2: IMPORTS
    # -------------------------------------------------------------------------
    print("\n[Test 2/6] Verifying Clean Module Imports...")
    import app.components.email_examples
    import app.components.prediction
    import app.components.visualization
    print("  [PASS] All component modules imported cleanly without side effects.")

    # -------------------------------------------------------------------------
    # TEST 3: PRESET VALIDATION (4 PRESETS x 2 MODELS = 8 COMBINATIONS)
    # -------------------------------------------------------------------------
    print("\n[Test 3/6] Running Mandatory Preset Classification Validation...")
    print("  Loading Pipeline B (Champion) and Pipeline C (Alternative)...")
    model_b = load_champion_model()
    model_c = load_pipeline_c_model()

    presets = get_email_examples()
    assert len(presets) >= 4, f"Expected at least 4 presets, got {len(presets)}"

    test_presets = ["ham_technical", "ham_scheduling", "spam_financial", "spam_phishing"]

    print("\n" + "-" * 90)
    print(f"| {'Preset':<22} | {'Model':<12} | {'Margin':>8} | {'Threshold':>10} | {'Prediction':<15} | {'Expected':<8} | {'Status':<6} |")
    print("-" * 90)

    preset_results = []
    for p_key in test_presets:
        preset_data = presets[p_key]
        expected_cat = preset_data["category"]
        text = preset_data["body"]

        # Test against Pipeline B
        res_b = classify_email(model_b, text, threshold=PIPELINE_B_CALIBRATED_THRESHOLD)
        actual_cat_b = "Spam" if res_b["is_spam"] else "Ham"
        status_b = "PASS" if actual_cat_b == expected_cat else "FAIL"
        assert status_b == "PASS", f"Preset {p_key} failed on Pipeline B: expected {expected_cat}, got {actual_cat_b}"
        preset_results.append((preset_data["name"], "Pipeline B", res_b["decision_margin"], PIPELINE_B_CALIBRATED_THRESHOLD, actual_cat_b, expected_cat, status_b))
        print(f"| {p_key:<22} | {'Pipeline B':<12} | {res_b['decision_margin']:>+8.4f} | {PIPELINE_B_CALIBRATED_THRESHOLD:>10.4f} | {actual_cat_b:<15} | {expected_cat:<8} | {status_b:<6} |")

        # Test against Pipeline C
        res_c = classify_email(model_c, text, threshold=PIPELINE_C_CALIBRATED_THRESHOLD)
        actual_cat_c = "Spam" if res_c["is_spam"] else "Ham"
        status_c = "PASS" if actual_cat_c == expected_cat else "FAIL"
        assert status_c == "PASS", f"Preset {p_key} failed on Pipeline C: expected {expected_cat}, got {actual_cat_c}"
        preset_results.append((preset_data["name"], "Pipeline C", res_c["decision_margin"], PIPELINE_C_CALIBRATED_THRESHOLD, actual_cat_c, expected_cat, status_c))
        print(f"| {p_key:<22} | {'Pipeline C':<12} | {res_c['decision_margin']:>+8.4f} | {PIPELINE_C_CALIBRATED_THRESHOLD:>10.4f} | {actual_cat_c:<15} | {expected_cat:<8} | {status_c:<6} |")

    print("-" * 90)
    print(f"  [PASS] All 8/8 preset classification combinations passed perfectly.")

    # -------------------------------------------------------------------------
    # TEST 4: VISUALIZATION COMPONENT
    # -------------------------------------------------------------------------
    print("\n[Test 4/6] Verifying Decision Boundary Visualization Gauge...")
    vis_cases = [
        {"margin": -2.5, "threshold": PIPELINE_B_CALIBRATED_THRESHOLD, "desc": "Negative margin (Strong Ham)"},
        {"margin": PIPELINE_B_CALIBRATED_THRESHOLD, "threshold": PIPELINE_B_CALIBRATED_THRESHOLD, "desc": "Margin exactly at threshold"},
        {"margin": 5.0, "threshold": PIPELINE_B_CALIBRATED_THRESHOLD, "desc": "Large positive margin (Extreme Spam)"},
        {"margin": -0.5, "threshold": PIPELINE_C_CALIBRATED_THRESHOLD, "desc": "Pipeline C threshold (Ham)"},
        {"margin": 1.2, "threshold": PIPELINE_C_CALIBRATED_THRESHOLD, "desc": "Pipeline C threshold (Spam)"},
    ]

    for i, case in enumerate(vis_cases, 1):
        svg = generate_margin_gauge_svg(case["margin"], case["threshold"], model_name="Test Model")
        assert "<svg" in svg and "</svg>" in svg, f"Case {i} SVG generation failed"
        assert "Continuous Decision Boundary Margin" in svg, "Missing title in SVG card"
        assert f"Threshold &tau;: {case['threshold']:+.4f}" in svg, "Missing threshold label in SVG"
        assert f"Margin: {case['margin']:+.4f}" in svg, "Missing margin label in SVG"
        print(f"  [PASS] Case {i}: {case['desc']} (margin={case['margin']:+.4f}, thresh={case['threshold']:+.4f}) -> SVG valid ({len(svg)} chars)")

    # -------------------------------------------------------------------------
    # TEST 5: PREDICTION COMPONENT
    # -------------------------------------------------------------------------
    print("\n[Test 5/6] Verifying Prediction Verdict Card & Heuristic Breakdown...")

    # 5.1 Spam Result Card
    spam_result = {
        "is_spam": True,
        "margin": 0.6050,
        "threshold": 0.0041,
        "latency_ms": 3.45,
        "model_name": "Pipeline B - Champion",
        "heuristics": None,
    }
    spam_html = generate_prediction_card_html(spam_result)
    assert "SECURITY ALERT: MALICIOUS SPAM DETECTED" in spam_html
    assert "+0.6050" in spam_html
    assert "+0.0041" in spam_html
    assert "+0.6009" in spam_html
    assert "3.45 ms" in spam_html
    print("  [PASS] Spam result card verified.")

    # 5.2 Ham Result Card
    ham_result = {
        "is_spam": False,
        "margin": -0.7818,
        "threshold": 0.0041,
        "latency_ms": 5.48,
        "model_name": "Pipeline B - Champion",
        "heuristics": None,
    }
    ham_html = generate_prediction_card_html(ham_result)
    assert "VERIFIED: LEGITIMATE COMMUNICATION (HAM)" in ham_html
    assert "-0.7818" in ham_html
    assert "-0.7859" in ham_html
    print("  [PASS] Ham result card verified.")

    # 5.3 Pipeline C Result with 6 Heuristic Features
    heuristics_data = extract_heuristic_features(presets["spam_phishing"]["body"])
    pipe_c_result = {
        "is_spam": True,
        "margin": 0.6860,
        "threshold": 0.0877,
        "latency_ms": 2.02,
        "model_name": "Pipeline C - High-Throughput",
        "heuristics": heuristics_data,
    }
    pipe_c_html = generate_prediction_card_html(pipe_c_result)
    assert "Structural Heuristic Breakdown (Pipeline C)" in pipe_c_html
    assert "Extracted via Pipeline C Domain Feature Transformer" in pipe_c_html
    assert "Uppercase Ratio" in pipe_c_html
    assert "ALL-CAPS Words" in pipe_c_html
    assert "Exclamation Marks" in pipe_c_html
    assert "Question Marks" in pipe_c_html
    assert "Currency Symbols" in pipe_c_html
    assert "URL Count" in pipe_c_html
    print("  [PASS] Pipeline C heuristic table verified.")

    # 5.4 Missing Heuristics & Defensive Inputs
    missing_h_result = {
        "is_spam": False,
        "margin": -0.5,
        "threshold": 0.0,
        "latency_ms": 1.5,
        "model_name": "Pipeline C",
        "heuristics": None,
    }
    missing_html = generate_prediction_card_html(missing_h_result)
    assert "Structural Heuristic Breakdown" not in missing_html
    print("  [PASS] Missing heuristics handled cleanly without error.")

    # 5.5 Invalid / Empty Inputs
    empty_html = generate_prediction_card_html({})
    assert "Diagnostic Notice" in empty_html
    invalid_html = generate_prediction_card_html({"margin": "invalid_num"})
    assert "Input Diagnostic Error" in invalid_html
    print("  [PASS] Invalid inputs handled gracefully with fallback diagnostics.")

    # -------------------------------------------------------------------------
    # TEST 6: INTEGRITY VERIFICATION
    # -------------------------------------------------------------------------
    print("\n[Test 6/6] Verifying Locked System Baselines & Integrity...")
    prep_path = REPO_ROOT / "src" / "preprocessing.py"
    prep_lines = len(prep_path.read_text(encoding="utf-8").splitlines())
    assert prep_lines == 487, f"src/preprocessing.py line count mismatch: expected 487, got {prep_lines}"
    print(f"  [PASS] src/preprocessing.py line count: {prep_lines} lines (EXACT)")

    raw_md5 = calc_md5(REPO_ROOT / "dataset" / "spam_assassin.csv")
    clean_md5 = calc_md5(REPO_ROOT / "dataset" / "spam_assassin_cleaned.csv")
    train_sha = calc_sha256(REPO_ROOT / "dataset" / "train_split_indices.npy")
    test_sha = calc_sha256(REPO_ROOT / "dataset" / "test_split_indices.npy")

    assert raw_md5 == "cc161a4efd9534c8a32664a52942d097", f"Raw dataset MD5 mismatch: {raw_md5}"
    assert clean_md5 == "87ab383a0a2853ff148a0e61fb5f2e5d", f"Cleaned dataset MD5 mismatch: {clean_md5}"
    assert train_sha == "84a9ef640ccc54711068d52c1f22d767aa800142c0f7b4822014855158e61d91", f"Train SHA mismatch: {train_sha}"
    assert test_sha == "bb1d408a89a7f53177defd1e4cd11d1036cccb577e3bc670a13c0d170210a0f5", f"Test SHA mismatch: {test_sha}"
    print("  [PASS] All dataset checksums verified bitwise:")
    print(f"         Raw dataset MD5:        {raw_md5}")
    print(f"         Cleaned dataset MD5:    {clean_md5}")
    print(f"         Train split SHA-256:    {train_sha[:24]}...")
    print(f"         Test split SHA-256:     {test_sha[:24]}... (UNTOUCHED)")

    print("\n" + "=" * 90)
    print("STAGE 7B AUDIT: ALL TESTS PASSED SUCCESSFULLY! (EXIT CODE 0)")
    print("=" * 90)


if __name__ == "__main__":
    test_stage7b()
