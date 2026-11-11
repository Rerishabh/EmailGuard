"""Prediction card and diagnostic summary component for Streamlit.

Renders an enterprise cybersecurity verdict banner, four key decision metrics
(Decision Margin, Operating Threshold, Boundary Delta, Inference Latency), and
an optional Pipeline C heuristic feature breakdown table.
"""

from __future__ import annotations

import html
import math
from typing import Any, Dict, Optional
import streamlit as st


def _format_heuristic_row(key: str, data: Any) -> tuple[str, str, str]:
    """Helper to parse and format a heuristic feature item into (name, display_value, description)."""
    default_meta = {
        "uppercase_ratio": ("Uppercase Ratio", "Ratio of uppercase letters to alphabetic characters."),
        "all_caps_word_count": ("ALL-CAPS Words", "Count of words with >= 3 consecutive uppercase characters."),
        "exclamation_count": ("Exclamation Marks", "Total count of '!' exclamation marks."),
        "question_count": ("Question Marks", "Total count of '?' question marks."),
        "currency_symbol_count": ("Currency Symbols", "Count of monetary currency symbols ($ € £ ¥)."),
        "url_count": ("URL Count", "Count of explicit web URLs or domain links."),
    }

    fallback_name, fallback_desc = default_meta.get(key, (key.replace("_", " ").title(), "Structural feature."))

    if isinstance(data, dict):
        name = data.get("name", fallback_name)
        desc = data.get("description", fallback_desc)
        formatted = data.get("formatted")
        if formatted is not None:
            return str(name), str(formatted), str(desc)
        val = data.get("value")
    else:
        name = fallback_name
        desc = fallback_desc
        val = data

    # Format numeric value
    if val is None:
        return str(name), "N/A", str(desc)
    try:
        val_f = float(val)
        if key == "uppercase_ratio":
            return str(name), f"{val_f:.2%}", str(desc)
        return str(name), str(int(round(val_f))), str(desc)
    except (ValueError, TypeError):
        return str(name), str(val), str(desc)


def generate_prediction_card_html(result: Dict[str, Any]) -> str:
    """Generate the HTML markup for the enterprise classification verdict card.

    Validates inputs defensively and outputs:
    1. Security verdict banner (Spam Alert vs Legitimate Verification).
    2. Four metric cards: Decision Margin, Operating Threshold, Boundary Delta, Latency.
    3. Pipeline C heuristic feature breakdown table (if heuristics are present) or Pipeline B architecture note.

    Returns:
        Renderable HTML string.
    """
    if not isinstance(result, dict) or not result:
        return """<div class="prediction-error-card" style="
            background: #F7E9E5; border: 1px solid #E8C5BE; border-left: 4px solid #C47770; border-radius: 6px;
            padding: 1rem 1.25rem; color: #34352F; font-family: Inter, system-ui, sans-serif; margin: 1rem 0;
        ">
            <strong style="color: #C47770;">Diagnostic Notice:</strong> No classification result dictionary provided.
        </div>"""

    # Extract & validate fields
    margin_raw = result.get("margin", result.get("decision_margin"))
    threshold_raw = result.get("threshold", 0.0)
    latency_raw = result.get("latency_ms", 0.0)
    model_name = str(result.get("model_name", "LinearSVC Classifier"))
    heuristics = result.get("heuristics")

    # Validate numeric margin
    if margin_raw is None:
        return """<div class="prediction-error-card" style="
            background: #F7E9E5; border: 1px solid #E8C5BE; border-left: 4px solid #C47770; border-radius: 6px;
            padding: 1rem 1.25rem; color: #34352F; font-family: Inter, system-ui, sans-serif; margin: 1rem 0;
        ">
            <strong style="color: #C47770;">Input Diagnostic Error:</strong> Missing decision margin in evaluation result.
        </div>"""

    try:
        margin = float(margin_raw)
        if not math.isfinite(margin):
            raise ValueError("Non-finite margin")
    except (ValueError, TypeError):
        return f"""<div class="prediction-error-card" style="
            background: #F7E9E5; border: 1px solid #E8C5BE; border-left: 4px solid #C47770; border-radius: 6px;
            padding: 1rem 1.25rem; color: #34352F; font-family: Inter, system-ui, sans-serif; margin: 1rem 0;
        ">
            <strong style="color: #C47770;">Input Diagnostic Error:</strong> Invalid numeric decision margin provided: <code>{html.escape(str(margin_raw))}</code>
        </div>"""

    try:
        threshold = float(threshold_raw) if threshold_raw is not None else 0.0
        if not math.isfinite(threshold):
            threshold = 0.0
    except (ValueError, TypeError):
        threshold = 0.0

    try:
        latency_ms = float(latency_raw) if latency_raw is not None else 0.0
        if not math.isfinite(latency_ms) or latency_ms < 0.0:
            latency_ms = 0.0
    except (ValueError, TypeError):
        latency_ms = 0.0

    # Decision rule: strictly margin >= threshold
    if "is_spam" in result:
        is_spam = bool(result["is_spam"])
    else:
        is_spam = margin >= threshold

    delta = margin - threshold

    # Verdict styles (Warm Minimalist cybersecurity palette)
    if is_spam:
        banner_title = "SECURITY ALERT: MALICIOUS SPAM DETECTED"
        banner_desc = "Email content exhibits strong statistical patterns characteristic of unsolicited or fraudulent messaging."
        banner_bg = "#F7E9E5"
        banner_border = "#E8C5BE"
        banner_color = "#C47770"
        banner_icon = "&#9888;"  # Warning symbol
        badge_bg = "#C47770"
        badge_text = "CLASSIFICATION: SPAM (MALICIOUS)"
    else:
        banner_title = "VERIFIED: LEGITIMATE COMMUNICATION (HAM)"
        banner_desc = "Email content aligns with verified benign communication patterns and remains safely below the alert threshold."
        banner_bg = "#EAF1E8"
        banner_border = "#C8DCBE"
        banner_color = "#64866C"
        banner_icon = "&#10004;"  # Checkmark
        badge_bg = "#64866C"
        badge_text = "CLASSIFICATION: HAM (LEGITIMATE)"

    # Metric Cards HTML
    metrics_html = f"""
    <div style="
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.85rem;
        margin-top: 1rem;
    ">
        <!-- Metric 1: Decision Margin -->
        <div style="background: #FFFCF6; border: 1px solid #E4DED2; border-radius: 6px; padding: 0.85rem 1rem; box-shadow: 0 1px 2px rgba(52,53,47,0.03);">
            <div style="font-size: 0.75rem; font-weight: 600; color: #89897F; text-transform: uppercase; letter-spacing: 0.04em;">
                Decision Margin
            </div>
            <div style="font-size: 1.45rem; font-weight: 700; color: #34352F; font-variant-numeric: tabular-nums; margin-top: 0.2rem;">
                {margin:+.4f}
            </div>
            <div style="font-size: 0.725rem; color: #6F7067; margin-top: 0.15rem;">
                Signed distance to hyperplane
            </div>
        </div>

        <!-- Metric 2: Operating Threshold -->
        <div style="background: #FFFCF6; border: 1px solid #E4DED2; border-radius: 6px; padding: 0.85rem 1rem; box-shadow: 0 1px 2px rgba(52,53,47,0.03);">
            <div style="font-size: 0.75rem; font-weight: 600; color: #89897F; text-transform: uppercase; letter-spacing: 0.04em;">
                Operating Threshold (&tau;)
            </div>
            <div style="font-size: 1.45rem; font-weight: 700; color: #34352F; font-variant-numeric: tabular-nums; margin-top: 0.2rem;">
                {threshold:+.4f}
            </div>
            <div style="font-size: 0.725rem; color: #6F7067; margin-top: 0.15rem;">
                Calibrated decision boundary
            </div>
        </div>

        <!-- Metric 3: Boundary Delta -->
        <div style="background: #FFFCF6; border: 1px solid #E4DED2; border-radius: 6px; padding: 0.85rem 1rem; box-shadow: 0 1px 2px rgba(52,53,47,0.03);">
            <div style="font-size: 0.75rem; font-weight: 600; color: #89897F; text-transform: uppercase; letter-spacing: 0.04em;">
                Boundary Delta (&Delta;)
            </div>
            <div style="font-size: 1.45rem; font-weight: 700; color: {banner_color}; font-variant-numeric: tabular-nums; margin-top: 0.2rem;">
                {delta:+.4f}
            </div>
            <div style="font-size: 0.725rem; color: #6F7067; margin-top: 0.15rem;">
                Margin minus threshold
            </div>
        </div>

        <!-- Metric 4: Inference Latency -->
        <div style="background: #FFFCF6; border: 1px solid #E4DED2; border-radius: 6px; padding: 0.85rem 1rem; box-shadow: 0 1px 2px rgba(52,53,47,0.03);">
            <div style="font-size: 0.75rem; font-weight: 600; color: #89897F; text-transform: uppercase; letter-spacing: 0.04em;">
                Inference Latency
            </div>
            <div style="font-size: 1.45rem; font-weight: 700; color: #34352F; font-variant-numeric: tabular-nums; margin-top: 0.2rem;">
                {latency_ms:.2f} ms
            </div>
            <div style="font-size: 0.725rem; color: #6F7067; margin-top: 0.15rem;">
                Single-email execution time
            </div>
        </div>
    </div>
    """

    # Heuristics Table HTML (only when heuristics are provided)
    if isinstance(heuristics, dict) and len(heuristics) > 0:
        rows = []
        for k, v in heuristics.items():
            name, disp, desc = _format_heuristic_row(k, v)
            rows.append(f"""
            <tr style="border-bottom: 1px solid #F0EDE5;">
                <td style="padding: 0.6rem 0.85rem; font-weight: 600; color: #34352F;">{name}</td>
                <td style="padding: 0.6rem 0.85rem; font-variant-numeric: tabular-nums; font-weight: 700; color: #34352F; text-align: right;">{disp}</td>
                <td style="padding: 0.6rem 0.85rem; color: #6F7067; font-size: 0.775rem;">{desc}</td>
            </tr>
            """)
        table_rows = "".join(rows)

        heuristics_html = f"""
        <div style="
            background: #FFFCF6;
            border: 1px solid #E4DED2;
            border-radius: 6px;
            padding: 1rem 1.25rem;
            margin-top: 1rem;
            box-shadow: 0 1px 2px rgba(52,53,47,0.03);
        ">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.75rem;">
                <h5 style="margin: 0; font-size: 0.925rem; font-weight: 600; color: #34352F;">
                    Structural Heuristic Breakdown (Pipeline C)
                </h5>
                <span style="font-size: 0.75rem; color: #89897F; font-style: italic;">
                    Extracted via Pipeline C Domain Feature Transformer
                </span>
            </div>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.825rem; text-align: left;">
                <thead>
                    <tr style="background: #F0EDE5; border-bottom: 1px solid #E4DED2;">
                        <th style="padding: 0.5rem 0.85rem; color: #6F7067; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em;">Feature</th>
                        <th style="padding: 0.5rem 0.85rem; color: #6F7067; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; text-align: right;">Observed Value</th>
                        <th style="padding: 0.5rem 0.85rem; color: #6F7067; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em;">Engineering Specification</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """
    else:
        heuristics_html = """
        <div style="
            margin-top: 1rem;
            padding: 0.65rem 0.85rem;
            background: #FFFCF6;
            border: 1px dashed #E4DED2;
            border-radius: 6px;
            font-size: 0.775rem;
            color: #6F7067;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        ">
            <span style="font-size: 0.85rem;">ℹ️</span>
            <span><strong>Pipeline Architecture Note:</strong> Pure NLP Pipeline: Heuristic features inactive for this model.</span>
        </div>
        """

    # Combined Card Container
    card_html = f"""
    <div class="prediction-card-wrapper" style="
        background: #FFFCF6;
        border: 1px solid #E4DED2;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 1.25rem 0;
        box-shadow: 0 1px 3px rgba(52,53,47,0.04);
        font-family: Inter, system-ui, -apple-system, sans-serif;
    ">
        <!-- Security Verdict Banner -->
        <div style="
            background: {banner_bg};
            border: 1px solid {banner_border};
            border-left: 4px solid {banner_color};
            border-radius: 6px;
            padding: 1rem 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.15rem; color: {banner_color};">{banner_icon}</span>
                    <h3 style="margin: 0; font-size: 1.1rem; font-weight: 700; color: {banner_color}; letter-spacing: -0.01em;">
                        {banner_title}
                    </h3>
                </div>
                <p style="margin: 0.35rem 0 0 1.65rem; font-size: 0.825rem; color: #34352F;">
                    {banner_desc}
                </p>
            </div>
            <div style="text-align: right; min-width: 170px;">
                <span style="
                    background: {badge_bg};
                    color: #FFFFFF;
                    font-size: 0.725rem;
                    font-weight: 700;
                    letter-spacing: 0.05em;
                    padding: 0.35rem 0.65rem;
                    border-radius: 4px;
                    display: inline-block;
                ">
                    {badge_text}
                </span>
                <div style="font-size: 0.725rem; color: #89897F; margin-top: 0.35rem;">
                    Evaluated by {model_name}
                </div>
            </div>
        </div>

        <!-- 4 Key Metrics -->
        {metrics_html}

        <!-- Heuristics Table (if present) / Architecture Note -->
        {heuristics_html}
    </div>
    """
    return "\n".join(line.strip() for line in card_html.splitlines() if line.strip())


def render_prediction_card(result: Dict[str, Any]) -> None:
    """Render the enterprise classification verdict card inside a Streamlit application.

    Gracefully renders via st.markdown(..., unsafe_allow_html=True).
    """
    html_markup = generate_prediction_card_html(result)
    st.markdown(html_markup, unsafe_allow_html=True)
