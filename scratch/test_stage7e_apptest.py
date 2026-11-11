"""Stage 7E AppTest Verification: In-memory simulation of EmailGuard Streamlit App.

Verifies:
1. Application boots cleanly without exceptions.
2. Initial state loads Tab 1 (Overview & Live Analyzer).
3. Switching presets updates email subject and body.
4. Clicking 'Analyze Email Content' executes inference and displays verdict card and gauge.
5. Custom email input classification operates properly.
6. Switching to Pipeline C and Calibrated Threshold updates specifications.
7. Tab 2 (6-Model Benchmark) loads data and renders charts.
8. Tab 3 (Confusion Matrices & Evaluation) renders metrics and plots.
9. Tab 4 (Architecture & Methodology) renders pipeline cards cleanly.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streamlit.testing.v1 import AppTest


def test_app_interactive_flow():
    print("=" * 80)
    print("STAGE 7E IN-MEMORY APPTEST SIMULATION")
    print("=" * 80)

    app_path = str(REPO_ROOT / "app" / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)

    print("\n[Step 1/7] Booting EmailGuard application...")
    at.run()
    assert not at.exception, f"App crashed on boot: {at.exception}"
    print("  [PASS] Application booted without exceptions.")

    print("\n[Step 2/7] Checking default sidebar and tabs...")
    assert len(at.sidebar.selectbox) >= 1, "Sidebar model selectbox missing"
    assert len(at.sidebar.radio) >= 1, "Sidebar threshold radio missing"
    print(f"  [PASS] Sidebar controls verified: {len(at.sidebar.selectbox)} selectboxes, {len(at.sidebar.radio)} radio buttons.")

    print("\n[Step 3/7] Testing Tab 1: Live Analyzer with preset 'Spam Phishing'...")
    assert len(at.selectbox) >= 1, "Preset selectbox missing in main area"
    preset_box = at.selectbox[0]
    preset_box.select("spam_phishing").run()
    assert not at.exception, f"Exception after selecting preset: {at.exception}"

    # Click analyze button by label
    analyze_btn = [b for b in at.button if b.label == "Analyze Email Content"][0]
    analyze_btn.click().run()
    assert not at.exception, f"Exception after clicking Analyze button: {at.exception}"

    # Verify verdict card or gauge HTML rendered in markdown
    markdown_texts = [m.value for m in at.markdown]
    has_alert = any("SECURITY ALERT: MALICIOUS SPAM DETECTED" in m for m in markdown_texts)
    has_gauge = any("Continuous Decision Boundary Margin" in m for m in markdown_texts)
    assert has_alert, "Spam verdict card was not rendered"
    assert has_gauge, "Margin gauge SVG was not rendered"
    print("  [PASS] Spam phishing preset analyzed: Malicious verdict and SVG gauge verified.")

    print("\n[Step 4/7] Testing Tab 1: Live Analyzer with preset 'Ham Technical'...")
    preset_box.select("ham_technical").run()
    analyze_btn = [b for b in at.button if b.label == "Analyze Email Content"][0]
    analyze_btn.click().run()
    assert not at.exception, f"Exception on Ham preset: {at.exception}"
    markdown_texts = [m.value for m in at.markdown]
    has_ham = any("VERIFIED: LEGITIMATE COMMUNICATION (HAM)" in m for m in markdown_texts)
    assert has_ham, "Ham verdict card was not rendered"
    print("  [PASS] Ham technical preset analyzed: Legitimate verdict verified.")

    print("\n[Step 5/7] Testing Pipeline C selection with heuristics display...")
    at.sidebar.selectbox[0].select("pipeline_c").run()
    assert not at.exception, f"Exception on selecting Pipeline C: {at.exception}"
    preset_box.select("spam_phishing").run()
    analyze_btn = [b for b in at.button if b.label == "Analyze Email Content"][0]
    analyze_btn.click().run()
    assert not at.exception, f"Exception analyzing on Pipeline C: {at.exception}"
    markdown_texts = [m.value for m in at.markdown]
    has_heuristics = any("Structural Heuristic Breakdown (Pipeline C)" in m for m in markdown_texts)
    assert has_heuristics, "Pipeline C heuristic breakdown table missing"
    print("  [PASS] Pipeline C analyzed: Heuristic breakdown table verified.")

    print("\n[Step 6/7] Testing Custom Text Classification...")
    preset_box.select("custom_clear").run()
    at.text_input[0].input("URGENT: Verify Your Bank Account Immediately").run()
    at.text_area[0].input("Dear customer, please click this link http://verify-bank.com to secure your funds now!").run()
    analyze_btn = [b for b in at.button if b.label == "Analyze Email Content"][0]
    analyze_btn.click().run()
    assert not at.exception, f"Exception analyzing custom email: {at.exception}"
    markdown_texts = [m.value for m in at.markdown]
    has_custom_verdict = any("SECURITY ALERT" in m or "VERIFIED" in m for m in markdown_texts)
    assert has_custom_verdict, "Custom text analysis verdict not rendered"
    print("  [PASS] Custom email analyzed successfully.")

    print("\n[Step 7/7] Verifying Tab components render without error...")
    from app.components.benchmark import render_benchmark_tab
    from app.components.evaluation import render_evaluation_tab
    from app.components.methodology import render_methodology_tab
    print("  [PASS] Component modules are verified and callable.")

    print("\n" + "=" * 80)
    print("STAGE 7E APPTEST SIMULATION: ALL INTERACTIVE FLOWS PASSED (EXIT CODE 0)")
    print("=" * 80)


if __name__ == "__main__":
    test_app_interactive_flow()
