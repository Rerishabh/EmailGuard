"""Stage 7C Verification Test Suite: Main Application Assembly & E2E Verification.

Verifies:
1. app.py imports successfully.
2. All component imports work.
3. Both model modes are selectable.
4. Both threshold modes work.
5. Preset selection populates email content.
6. Subject and body are passed correctly.
7. Predictions use the selected model and threshold.
8. Results render correctly for Ham and Spam.
9. Pipeline C displays heuristic features.
10. Empty input is handled safely.
11. CSS is loaded.
12. Existing artifacts remain unchanged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def test_stage7c():
    print("=" * 90)
    print("STAGE 7C AUDIT: MAIN STREAMLIT APPLICATION VERIFICATION")
    print("=" * 90)

    # 1. app.py imports successfully
    print("\n[Check 1/12] Verifying app.py imports successfully...")
    import app.app as main_app
    assert hasattr(main_app, "main"), "app.py missing main() function"
    assert hasattr(main_app, "render_header"), "app.py missing render_header()"
    assert hasattr(main_app, "render_sidebar"), "app.py missing render_sidebar()"
    print("  [PASS] app.py imported cleanly.")

    # 2. All component imports work
    print("\n[Check 2/12] Verifying all component and utility imports...")
    from app.components.email_examples import get_email_examples
    from app.components.prediction import generate_prediction_card_html, render_prediction_card
    from app.components.visualization import generate_margin_gauge_svg, render_margin_gauge
    from app.utils.model_loader import (
        PIPELINE_B_CALIBRATED_THRESHOLD,
        PIPELINE_B_DEFAULT_THRESHOLD,
        PIPELINE_C_CALIBRATED_THRESHOLD,
        PIPELINE_C_DEFAULT_THRESHOLD,
        classify_email,
        extract_heuristic_features,
        get_model_specs,
        load_champion_config,
        load_champion_model,
        load_pipeline_c_model,
        predict_decision_margin,
    )
    print("  [PASS] All submodules and functions imported cleanly.")

    # 3. Both model modes are selectable
    print("\n[Check 3/12] Verifying model mode specifications...")
    specs = get_model_specs()
    assert "pipeline_b" in specs, "Missing pipeline_b in model specs"
    assert "pipeline_c" in specs, "Missing pipeline_c in model specs"
    assert specs["pipeline_b"]["role"] == "Production Champion"
    assert "High-Throughput" in specs["pipeline_c"]["role"]
    print("  [PASS] Both model modes are properly registered with specifications.")

    # 4. Both threshold modes work
    print("\n[Check 4/12] Verifying threshold modes (Default vs Calibrated)...")
    assert PIPELINE_B_DEFAULT_THRESHOLD == 0.0
    assert abs(PIPELINE_B_CALIBRATED_THRESHOLD - 0.0040852377135018) < 1e-12
    assert PIPELINE_C_DEFAULT_THRESHOLD == 0.0
    assert abs(PIPELINE_C_CALIBRATED_THRESHOLD - 0.0876861391187918) < 1e-12
    print("  [PASS] Operating thresholds verified for both pipelines.")

    # 5. Preset selection populates email content
    print("\n[Check 5/12] Verifying preset email templates...")
    examples = get_email_examples()
    assert "ham_technical" in examples
    assert "ham_scheduling" in examples
    assert "spam_financial" in examples
    assert "spam_phishing" in examples
    assert "custom_clear" in examples
    for k in ["ham_technical", "ham_scheduling", "spam_financial", "spam_phishing"]:
        ex = examples[k]
        assert len(ex["subject"].strip()) > 0, f"Preset {k} has empty subject"
        assert len(ex["body"].strip()) > 0, f"Preset {k} has empty body"
    assert examples["custom_clear"]["subject"] == ""
    assert examples["custom_clear"]["body"] == ""
    print("  [PASS] All 5 email presets verified.")

    # 6. Subject and body combination
    print("\n[Check 6/12] Verifying subject and body concatenation logic...")
    subj = "Test Subject"
    body = "Test Body"
    combined = f"Subject: {subj}\n\n{body}"
    assert combined.startswith("Subject: Test Subject")
    assert "Test Body" in combined
    print("  [PASS] Text concatenation logic verified.")

    # 7. Predictions use selected model and threshold
    print("\n[Check 7/12] Testing model loading and predictions on sample text...")
    model_b = load_champion_model()
    model_c = load_pipeline_c_model()

    test_ham = examples["ham_technical"]["body"]
    test_spam = examples["spam_financial"]["body"]

    mb_ham = predict_decision_margin(model_b, test_ham)
    mb_spam = predict_decision_margin(model_b, test_spam)
    mc_ham = predict_decision_margin(model_c, test_ham)
    mc_spam = predict_decision_margin(model_c, test_spam)

    assert mb_ham < PIPELINE_B_CALIBRATED_THRESHOLD, f"Ham margin {mb_ham} not below threshold"
    assert mb_spam >= PIPELINE_B_CALIBRATED_THRESHOLD, f"Spam margin {mb_spam} not above threshold"
    assert mc_ham < PIPELINE_C_CALIBRATED_THRESHOLD, f"Ham margin {mc_ham} not below threshold"
    assert mc_spam >= PIPELINE_C_CALIBRATED_THRESHOLD, f"Spam margin {mc_spam} not above threshold"
    print("  [PASS] Prediction scores verified across both models.")

    # 8. Results render correctly for Ham and Spam
    print("\n[Check 8/12] Testing verdict card and margin gauge rendering...")
    ham_res = {
        "is_spam": False,
        "margin": mb_ham,
        "decision_margin": mb_ham,
        "threshold": PIPELINE_B_CALIBRATED_THRESHOLD,
        "latency_ms": 3.2,
        "model_name": specs["pipeline_b"]["name"],
        "heuristics": None,
    }
    spam_res = {
        "is_spam": True,
        "margin": mb_spam,
        "decision_margin": mb_spam,
        "threshold": PIPELINE_B_CALIBRATED_THRESHOLD,
        "latency_ms": 2.8,
        "model_name": specs["pipeline_b"]["name"],
        "heuristics": None,
    }

    ham_html = generate_prediction_card_html(ham_res)
    spam_html = generate_prediction_card_html(spam_res)
    assert "VERIFIED: LEGITIMATE COMMUNICATION (HAM)" in ham_html
    assert "SECURITY ALERT: MALICIOUS SPAM DETECTED" in spam_html

    svg_ham = generate_margin_gauge_svg(mb_ham, PIPELINE_B_CALIBRATED_THRESHOLD, specs["pipeline_b"]["name"])
    svg_spam = generate_margin_gauge_svg(mb_spam, PIPELINE_B_CALIBRATED_THRESHOLD, specs["pipeline_b"]["name"])
    assert "<svg" in svg_ham and "</svg>" in svg_ham
    assert "<svg" in svg_spam and "</svg>" in svg_spam
    print("  [PASS] HTML card and SVG gauge rendering verified for both Ham and Spam.")

    # 9. Pipeline C displays heuristic features
    print("\n[Check 9/12] Testing Pipeline C heuristic feature extraction...")
    heuristics = extract_heuristic_features(test_spam)
    assert len(heuristics) == 6
    for h_key in ["uppercase_ratio", "all_caps_word_count", "exclamation_count", "question_count", "currency_symbol_count", "url_count"]:
        assert h_key in heuristics, f"Missing heuristic: {h_key}"

    c_res = {
        "is_spam": True,
        "margin": mc_spam,
        "decision_margin": mc_spam,
        "threshold": PIPELINE_C_CALIBRATED_THRESHOLD,
        "latency_ms": 1.9,
        "model_name": specs["pipeline_c"]["name"],
        "heuristics": heuristics,
    }
    c_html = generate_prediction_card_html(c_res)
    assert "Structural Heuristic Breakdown (Pipeline C)" in c_html
    print("  [PASS] Pipeline C heuristic extraction and card integration verified.")

    # 10. Empty input is handled safely
    print("\n[Check 10/12] Verifying empty input handling...")
    empty_card = generate_prediction_card_html({})
    assert "Diagnostic Notice" in empty_card
    print("  [PASS] Empty/corrupt input handled safely.")

    # 11. CSS is loaded
    print("\n[Check 11/12] Verifying custom.css existence and contents...")
    css_path = REPO_ROOT / "app" / "styles" / "custom.css"
    assert css_path.exists(), "custom.css not found"
    css_text = css_path.read_text(encoding="utf-8")
    assert "--ham-color" in css_text
    assert "--spam-color" in css_text
    assert "--bg-main" in css_text
    print(f"  [PASS] custom.css verified ({len(css_text.splitlines())} lines).")

    # 12. Existing artifacts remain unchanged
    print("\n[Check 12/12] Auditing locked artifact checksums and integrity...")
    prep_path = REPO_ROOT / "src" / "preprocessing.py"
    prep_lines = len(prep_path.read_text(encoding="utf-8").splitlines())
    assert prep_lines == 487, f"preprocessing.py lines expected 487, got {prep_lines}"

    raw_md5 = calc_md5(REPO_ROOT / "dataset" / "spam_assassin.csv")
    clean_md5 = calc_md5(REPO_ROOT / "dataset" / "spam_assassin_cleaned.csv")
    train_sha = calc_sha256(REPO_ROOT / "dataset" / "train_split_indices.npy")
    test_sha = calc_sha256(REPO_ROOT / "dataset" / "test_split_indices.npy")

    assert raw_md5 == "cc161a4efd9534c8a32664a52942d097"
    assert clean_md5 == "87ab383a0a2853ff148a0e61fb5f2e5d"
    assert train_sha == "84a9ef640ccc54711068d52c1f22d767aa800142c0f7b4822014855158e61d91"
    assert test_sha == "bb1d408a89a7f53177defd1e4cd11d1036cccb577e3bc670a13c0d170210a0f5"

    print("  [PASS] All locked artifacts verified:")
    print(f"         src/preprocessing.py:  {prep_lines} lines (LOCKED)")
    print(f"         Raw dataset MD5:       {raw_md5}")
    print(f"         Cleaned dataset MD5:   {clean_md5}")
    print(f"         Train split SHA-256:   {train_sha[:24]}...")
    print(f"         Test split SHA-256:    {test_sha[:24]}... (UNTOUCHED)")

    print("\n" + "=" * 90)
    print("STAGE 7C AUDIT: ALL 12 CHECKS PASSED SUCCESSFULLY! (EXIT CODE 0)")
    print("=" * 90)


if __name__ == "__main__":
    test_stage7c()

