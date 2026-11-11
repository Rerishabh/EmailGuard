from __future__ import annotations

import importlib
from pathlib import Path
import sys
import textwrap
import time
from typing import Any, Dict

# Ensure both the app directory and repository root are registered in sys.path
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st


def _load_module(name: str) -> Any:
    """Resolve an app submodule under either Streamlit sibling imports or package imports."""
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError):
        return importlib.import_module(f"app.{name}")


@st.cache_data(show_spinner=False)
def _read_custom_css(css_path_str: str) -> str:
    """Read custom.css once and cache across reruns."""
    return Path(css_path_str).read_text(encoding="utf-8")


def load_custom_css() -> None:
    """Inject enterprise cybersecurity design system CSS into the Streamlit DOM."""
    css_path = APP_DIR / "styles" / "custom.css"
    if css_path.exists():
        css_content = _read_custom_css(str(css_path))
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize session state defaults for email presets and form inputs."""
    if "subject_input" not in st.session_state:
        st.session_state["subject_input"] = ""
    if "body_input" not in st.session_state:
        st.session_state["body_input"] = ""
    if "preset_selector" not in st.session_state:
        st.session_state["preset_selector"] = "custom_clear"
    if "last_analyzed_result" not in st.session_state:
        st.session_state["last_analyzed_result"] = None


def on_preset_change() -> None:
    """Callback when a template preset is selected in the UI."""
    examples = _load_module("components.email_examples").get_email_examples()
    selected_key = st.session_state.get("preset_selector", "custom_clear")
    if selected_key in examples:
        preset = examples[selected_key]
        st.session_state["subject_input"] = preset["subject"]
        st.session_state["body_input"] = preset["body"]


def clear_form() -> None:
    """Reset form fields to blank custom input."""
    st.session_state["preset_selector"] = "custom_clear"
    st.session_state["subject_input"] = ""
    st.session_state["body_input"] = ""
    st.session_state["last_analyzed_result"] = None


def render_header() -> None:
    """Render the enterprise application header with SVG shield logo and 4 performance metrics."""
    shield_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="46" height="46" viewBox="0 0 48 48" '
        'aria-label="EmailGuard Shield Logo" style="display: block; flex-shrink: 0;">'
        '<path d="M24 4L7 11.5V22.5C7 33.1 14.3 42.6 24 45C33.7 42.6 41 33.1 41 22.5V11.5L24 4Z" '
        'fill="#3D5A73" stroke="#2D4459" stroke-width="1.5" stroke-linejoin="round"/>'
        '<path d="M24 7.5L10.5 13.5V22.5C10.5 31.1 16.3 39 24 41.2C31.7 39 37.5 31.1 37.5 22.5V13.5L24 7.5Z" '
        'fill="#EAF1E8" fill-opacity="0.16" stroke="#718C83" stroke-width="1.25" stroke-linejoin="round"/>'
        '<path d="M18.2 24.4L22.1 28.3L30.2 19.8" fill="none" stroke="#FFFCF6" stroke-width="3" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>'
    )
    brand_lines = [
        '<div class="emailguard-header-card" style="background: #FFFCF6; border: 1px solid #E4DED2; border-radius: 8px; padding: 1rem 1.35rem; margin-bottom: 0.85rem; box-shadow: 0 1px 3px rgba(52,53,47,0.05); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.85rem;">',
        '<div style="display: flex; align-items: center; gap: 0.95rem; min-width: 260px;">',
        shield_svg,
        '<div style="display: flex; flex-direction: column;">',
        '<div style="display: flex; align-items: center; flex-wrap: wrap; gap: 0.6rem;">',
        '<h1 class="emailguard-brand-title" style="margin: 0; font-size: 27px; font-weight: 700; color: #34352F; line-height: 1.18; letter-spacing: -0.02em;">EmailGuard</h1>',
        '<span style="background: #F0EDE5; border: 1px solid #E4DED2; color: #5E786F; font-size: 0.72rem; font-weight: 700; padding: 0.2rem 0.55rem; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.04em; line-height: 1.2;">PRODUCTION V1.0</span>',
        '</div>',
        '<p class="emailguard-brand-subtitle" style="margin: 0.25rem 0 0 0; color: #6F7067; font-size: 16px; font-weight: 500; line-height: 1.35;">Intelligent Email Spam Detection Using NLP</p>',
        '</div>',
        '</div>',
        '<div style="display: inline-flex; align-items: center; gap: 0.45rem; background: #EAF1E8; border: 1px solid #C8DCBE; color: #64866C; font-size: 0.75rem; font-weight: 700; padding: 0.35rem 0.75rem; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.04em; flex-shrink: 0;">',
        '<span style="width: 8px; height: 8px; background-color: #64866C; border-radius: 50%; display: inline-block;"></span>',
        '<span>MODEL LOADED &amp; ACTIVE</span>',
        '</div>',
        '</div>',
    ]
    st.markdown("\n".join(brand_lines), unsafe_allow_html=True)

    # B. Four Compact Metric Cards (Warm Minimalist Palette)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dataset Size", "3,201 Emails", "SpamAssassin Benchmark")
    m2.metric("Champion Model", "Pipeline B + LinearSVC", "Dual Hinge (C=2.0)")
    m3.metric("Test Accuracy", "99.06%", "Held-out Test (N=641)")
    m4.metric("False Positive Rate", "0.37%", "Strict FPR ≤ 0.50%")


def render_sidebar() -> tuple[str, bool]:
    """Render the sidebar controls for model architecture, threshold selection, and metadata."""
    sidebar_title = textwrap.dedent("""
    <div style="padding-bottom: 0.75rem; margin-bottom: 1.25rem; border-bottom: 1px solid #E4DED2;">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.25rem;">⚙️</span>
            <h3 style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #34352F; letter-spacing: -0.015em;">Security Engine</h3>
        </div>
        <p style="margin: 0.3rem 0 0 0; font-size: 0.84rem; font-weight: 500; color: #6F7067; line-height: 1.4;">
            Select classification topology and decision threshold policy.
        </p>
    </div>
    """).strip()
    st.sidebar.markdown(sidebar_title, unsafe_allow_html=True)

    specs = _load_module("utils.model_loader").get_model_specs()

    # 1. Model Selection
    model_options = {
        "pipeline_b": "Pipeline B: Production Champion",
        "pipeline_c": "Pipeline C: High-Throughput Alternative",
    }
    selected_model_key = st.sidebar.selectbox(
        "Classification Pipeline",
        options=list(model_options.keys()),
        format_func=lambda k: model_options[k],
        index=0,
        help="Pipeline B balances maximum precision and generalization. Pipeline C offers 2.72x faster inference.",
    )

    # 2. Threshold Policy Selection
    threshold_options = {
        "calibrated": "Calibrated Operating Threshold (Recommended)",
        "default": "Default Boundary Threshold (0.0)",
    }
    selected_threshold_key = st.sidebar.radio(
        "Operating Threshold Policy",
        options=list(threshold_options.keys()),
        format_func=lambda k: threshold_options[k],
        index=0,
        help="Calibrated thresholds were empirically determined during Stage 6C cross-validation to satisfy FPR <= 0.5%.",
    )
    is_calibrated = selected_threshold_key == "calibrated"

    # Resolve active threshold float
    active_spec = specs[selected_model_key]
    active_threshold = (
        active_spec["calibrated_threshold"] if is_calibrated else active_spec["default_threshold"]
    )

    # 3. Model Information Card (Warm Cream Palette)
    heur_bg = "#EAF1E8" if active_spec["uses_heuristics"] else "#F0EDE5"
    heur_color = "#64866C" if active_spec["uses_heuristics"] else "#89897F"
    heur_text = "Active (6 Features)" if active_spec["uses_heuristics"] else "Inactive (Pure NLP)"
    thresh_type = "Calibrated" if is_calibrated else "Default"

    spec_html = textwrap.dedent(f"""
    <div style="
        background: #FFFCF6; border: 1px solid #E4DED2; border-radius: 6px;
        padding: 0.85rem 1rem; margin-top: 1.25rem; box-shadow: 0 1px 2px rgba(52,53,47,0.03);
        font-size: 0.8rem;
    ">
        <div style="font-size: 0.7rem; font-weight: 700; color: #89897F; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
            Active Engine Specifications
        </div>
        <div style="line-height: 1.5;">
            <div style="margin-bottom: 0.4rem;">
                <span style="color: #89897F; display: block; font-size: 0.725rem;">TOPOLOGY</span>
                <strong style="color: #34352F;">{active_spec['name']}</strong>
            </div>
            <div style="margin-bottom: 0.4rem;">
                <span style="color: #89897F; display: block; font-size: 0.725rem;">CLASSIFIER</span>
                <span style="color: #34352F; font-weight: 600;">LinearSVC (Dual Hinge)</span>
            </div>
            <div style="margin-bottom: 0.4rem; font-variant-numeric: tabular-nums;">
                <span style="color: #89897F; display: block; font-size: 0.725rem;">OPERATING THRESHOLD (&tau;)</span>
                <strong style="color: #718C83;">{active_threshold:+.4f}</strong>
                <span style="color: #89897F; font-size: 0.725rem; margin-left: 0.25rem;">({thresh_type})</span>
            </div>
            <div style="margin-bottom: 0.4rem; font-variant-numeric: tabular-nums;">
                <span style="color: #89897F; display: block; font-size: 0.725rem;">BENCHMARK LATENCY</span>
                <strong style="color: #34352F;">{active_spec['latency_ms']:.2f} ms/email</strong>
            </div>
            <div>
                <span style="color: #89897F; display: block; font-size: 0.725rem; margin-bottom: 0.2rem;">HEURISTICS MODULE</span>
                <span style="
                    background: {heur_bg};
                    color: {heur_color};
                    padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.725rem;
                    display: inline-block;
                ">{heur_text}</span>
            </div>
        </div>
    </div>
    """).strip()
    st.sidebar.markdown(spec_html, unsafe_allow_html=True)

    # 4. Architectural Note
    contract_html = textwrap.dedent("""
    <div style="margin-top: 1.25rem; padding: 0.85rem; background: #F0EDE5; border: 1px dashed #E4DED2; border-radius: 6px; font-size: 0.775rem; color: #6F7067; line-height: 1.45;">
        <strong style="color: #34352F; display: block; margin-bottom: 0.25rem;">Decision Margin Contract:</strong>
        LinearSVC produces a continuous signed distance to the hyperplane. Scores at or above the operating threshold trigger a malicious spam alert.
    </div>
    """).strip()
    st.sidebar.markdown(contract_html, unsafe_allow_html=True)

    return selected_model_key, is_calibrated


def render_live_analyzer(selected_model_key: str, is_calibrated: bool) -> None:
    """Render the interactive live email analyzer section (Component C of Tab 1)."""
    model_loader = _load_module("utils.model_loader")
    specs = model_loader.get_model_specs()
    active_spec = specs[selected_model_key]
    active_threshold = (
        active_spec["calibrated_threshold"] if is_calibrated else active_spec["default_threshold"]
    )

    examples = _load_module("components.email_examples").get_email_examples()
    preset_keys = list(examples.keys())

    with st.container(border=True):
        st.subheader("Live Email Analyzer")
        st.write(
            "Select an audited email preset or enter custom email headers and text for real-time threat analysis:"
        )

        # 1. Preset Selector and Clear Form
        col_preset, col_clear = st.columns([5, 1])
        with col_preset:
            st.selectbox(
                "Select Realistic Email Template Preset:",
                options=preset_keys,
                format_func=lambda k: examples[k]["name"],
                key="preset_selector",
                on_change=on_preset_change,
                help="Audited email presets validated across both ML candidate pipelines.",
            )
        with col_clear:
            st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
            st.button("Clear Form", on_click=clear_form, use_container_width=True)

        # 2. Email Subject & Body
        st.text_input(
            "Email Subject",
            key="subject_input",
            placeholder="Enter email subject line (e.g., URGENT: Account Verification Required)...",
        )

        st.text_area(
            "Email Body",
            key="body_input",
            height=200,
            placeholder="Paste complete raw email message body here for deep NLP inspection...",
        )

        # 3. Action Button
        col_btn, _ = st.columns([2, 5])
        with col_btn:
            analyze_clicked = st.button("Analyze Email Content", type="primary", use_container_width=True)

    # Execution workflow
    if analyze_clicked:
        subject = st.session_state.get("subject_input", "").strip()
        body = st.session_state.get("body_input", "").strip()

        if not subject and not body:
            st.error("Please provide an email subject line or body text before running analysis.")
            st.session_state["last_analyzed_result"] = None
            return

        # Prepare raw combined email text for pipeline
        if subject and body:
            combined_text = f"Subject: {subject}\n\n{body}"
        elif body:
            combined_text = body
        else:
            combined_text = f"Subject: {subject}"

        # Load appropriate model
        try:
            with st.spinner(f"Loading {active_spec['name']}..."):
                if selected_model_key == "pipeline_b":
                    model = model_loader.load_champion_model()
                else:
                    model = model_loader.load_pipeline_c_model()
        except Exception as e:
            st.error(f"Failed to load model pipeline: {e}")
            st.session_state["last_analyzed_result"] = None
            return

        # Inference and timing
        try:
            t0 = time.perf_counter()
            margin = model_loader.predict_decision_margin(model, combined_text)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            is_spam = bool(margin >= active_threshold)

            # Heuristic features for Pipeline C
            heuristics = None
            if active_spec["uses_heuristics"]:
                heuristics = model_loader.extract_heuristic_features(combined_text)

            result: Dict[str, Any] = {
                "is_spam": is_spam,
                "margin": margin,
                "decision_margin": margin,
                "threshold": active_threshold,
                "latency_ms": latency_ms,
                "model_name": active_spec["name"],
                "heuristics": heuristics,
                "subject": subject,
            }
            st.session_state["last_analyzed_result"] = result
        except Exception as e:
            st.error(f"Inference computation error: {e}")
            st.session_state["last_analyzed_result"] = None
            return

    # Display Results if available
    current_result = st.session_state.get("last_analyzed_result")
    if current_result is not None:
        st.subheader("Classification Verdict & Diagnostic Output")

        # 1. Prediction Verdict Card & Metrics
        _load_module("components.prediction").render_prediction_card(current_result)

        # 2. Decision Boundary Continuum Gauge
        _load_module("components.visualization").render_margin_gauge(
            margin=current_result["decision_margin"],
            threshold=current_result["threshold"],
            model_name=current_result["model_name"],
        )


def main() -> None:
    """Main application routine."""
    # st.set_page_config must be the very first Streamlit command executed
    st.set_page_config(
        page_title="EmailGuard | Enterprise Email Security",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject styling and initialize session state
    load_custom_css()
    init_session_state()

    # Render Header & Sidebar
    render_header()
    selected_model_key, is_calibrated = render_sidebar()

    # Four-Tab Navigation Layout
    tab_overview, tab_benchmark, tab_evaluation, tab_methodology = st.tabs([
        "Overview & Live Analyzer",
        "6-Model Benchmark",
        "Confusion Matrices & Evaluation",
        "Architecture & Methodology",
    ])

    with tab_overview:
        # C. Live Email Analyzer (directly below metric cards!)
        render_live_analyzer(selected_model_key, is_calibrated)
        # D. Compact Dataset Summary (below analyzer!)
        _load_module("components.overview").render_dataset_summary()

    with tab_benchmark:
        _load_module("components.benchmark").render_benchmark_tab()

    with tab_evaluation:
        _load_module("components.evaluation").render_evaluation_tab()

    with tab_methodology:
        _load_module("components.methodology").render_methodology_tab()


if __name__ == "__main__":
    main()
