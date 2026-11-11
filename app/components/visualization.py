"""Continuous decision margin visualization component for Streamlit.

Renders an enterprise cybersecurity decision boundary gauge illustrating the signed
decision score produced by LinearSVC relative to the active operating threshold.
"""

from __future__ import annotations

import math
from typing import Optional
import streamlit as st


def generate_margin_gauge_svg(
    margin: float,
    threshold: float,
    model_name: str = "LinearSVC",
) -> str:
    """Generate a responsive, pixel-perfect SVG visualization of the decision margin.

    Illustrates:
    - Left region (Green): Legitimate Email / Ham (margin < threshold)
    - Right region (Red): Malicious Spam (margin >= threshold)
    - Threshold marker: The operating decision boundary (tau)
    - Margin marker: Current email signed score with boundary delta callout
    - Dynamic scaling centered around threshold, avoiding extreme value clipping.

    Returns:
        Raw HTML/SVG string.
    """
    # Defensive casting
    try:
        margin_f = float(margin)
        threshold_f = float(threshold)
    except (ValueError, TypeError):
        margin_f = 0.0
        threshold_f = 0.0

    if not math.isfinite(margin_f):
        margin_f = 0.0
    if not math.isfinite(threshold_f):
        threshold_f = 0.0

    # Decision rule
    is_spam = margin_f >= threshold_f
    delta = margin_f - threshold_f
    distance = abs(delta)

    # Dynamic coordinate range calculation
    span = max(distance * 1.6, 2.5)
    min_val = min(threshold_f - span, margin_f - 0.6)
    max_val = max(threshold_f + span, margin_f + 0.6)
    total_val_range = max_val - min_val
    if total_val_range <= 1e-9:
        total_val_range = 1.0

    # SVG layout geometry
    svg_width = 840
    svg_height = 230
    pad_left = 60
    pad_right = 60
    bar_width = svg_width - pad_left - pad_right
    bar_y = 95
    bar_height = 36

    def val_to_x(val: float) -> float:
        norm = (val - min_val) / total_val_range
        # Clamp to bar boundaries to prevent visual escaping
        norm = max(0.0, min(1.0, norm))
        return pad_left + norm * bar_width

    x_thresh = val_to_x(threshold_f)
    x_margin = val_to_x(margin_f)

    # Calculate tick marks for the scale
    step = 1.0
    if total_val_range > 10.0:
        step = 2.0
    elif total_val_range < 3.0:
        step = 0.5

    first_tick = math.ceil(min_val / step) * step
    ticks = []
    curr = first_tick
    while curr <= max_val:
        # Avoid ticks too close to the boundary marker
        xt = val_to_x(curr)
        if abs(xt - x_thresh) > 28 and pad_left + 15 <= xt <= pad_left + bar_width - 15:
            ticks.append((curr, xt))
        curr += step

    # Colors (Warm Minimalist palette)
    status_color = "#C47770" if is_spam else "#64866C"
    status_label = "MALICIOUS SPAM" if is_spam else "LEGITIMATE HAM"
    zone_class_label = "Spam Zone (Score >= Threshold)" if is_spam else "Ham Zone (Score < Threshold)"

    # SVG elements
    tick_svg_elements = []
    for val, xt in ticks:
        tick_svg_elements.append(
            f'<line x1="{xt:.1f}" y1="{bar_y + bar_height}" x2="{xt:.1f}" y2="{bar_y + bar_height + 6}" '
            f'stroke="#C8C4BA" stroke-width="1.2" />'
            f'<text x="{xt:.1f}" y="{bar_y + bar_height + 18}" font-family="Inter, system-ui, sans-serif" '
            f'font-size="11" fill="#6F7067" text-anchor="middle" font-weight="500">{val:+.1f}</text>'
        )
    ticks_markup = "\n    ".join(tick_svg_elements)

    # Offset badges if threshold and margin are close
    is_overlapping = abs(x_thresh - x_margin) < 70
    thresh_badge_y = 35 if is_overlapping else 42
    margin_badge_y = 188 if is_overlapping else 176

    svg = f"""<div class="margin-gauge-container" style="
    background: #FFFCF6;
    border: 1px solid #E4DED2;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(52,53,47,0.04);
    margin: 1.25rem 0;
">
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem;">
        <div>
            <h4 style="margin: 0; color: #34352F; font-size: 1.05rem; font-weight: 600; letter-spacing: -0.01em;">
                Continuous Decision Boundary Margin
            </h4>
            <p style="margin: 0.25rem 0 0 0; color: #6F7067; font-size: 0.825rem;">
                Signed decision score returned by LinearSVC. Classification is determined by the selected operating threshold.
            </p>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 0.75rem; color: #89897F; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Active Model</span>
            <div style="font-size: 0.875rem; font-weight: 600; color: #34352F;">{model_name}</div>
        </div>
    </div>

    <svg viewBox="0 0 {svg_width} {svg_height}" width="100%" height="auto" style="overflow: visible; display: block;">
        <defs>
            <linearGradient id="hamGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#EAF1E8" />
                <stop offset="100%" stop-color="#D6E4D4" />
            </linearGradient>
            <linearGradient id="spamGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#F7E9E5" />
                <stop offset="100%" stop-color="#EFCFC8" />
            </linearGradient>
            <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="1.5" stdDeviation="2" flood-color="#34352F" flood-opacity="0.18" />
            </filter>
        </defs>

        <!-- Base Background track -->
        <rect x="{pad_left}" y="{bar_y}" width="{bar_width}" height="{bar_height}" rx="5" fill="#F0EDE5" />

        <!-- Green Region: Ham / Legitimate (margin < threshold) -->
        <rect x="{pad_left}" y="{bar_y}" width="{max(0.0, x_thresh - pad_left):.1f}" height="{bar_height}" 
              rx="5" fill="url(#hamGradient)" stroke="#C8DCBE" stroke-width="1" />

        <!-- Red Region: Spam / Malicious (margin >= threshold) -->
        <rect x="{x_thresh:.1f}" y="{bar_y}" width="{max(0.0, pad_left + bar_width - x_thresh):.1f}" height="{bar_height}" 
              rx="5" fill="url(#spamGradient)" stroke="#E8C5BE" stroke-width="1" />

        <!-- Zone Labels inside Bar -->
        <text x="{pad_left + 16}" y="{bar_y + 22}" font-family="Inter, system-ui, sans-serif" font-size="11.5" 
              font-weight="600" fill="#64866C" letter-spacing="0.04em">LEGITIMATE HAM ZONE (&lt; &tau;)</text>
        <text x="{pad_left + bar_width - 16}" y="{bar_y + 22}" font-family="Inter, system-ui, sans-serif" font-size="11.5" 
              font-weight="600" fill="#C47770" text-anchor="end" letter-spacing="0.04em">MALICIOUS SPAM ZONE (&ge; &tau;)</text>

        <!-- Tick Marks & Numerical Scale -->
        {ticks_markup}

        <!-- Decision Boundary (Operating Threshold) Marker -->
        <line x1="{x_thresh:.1f}" y1="{bar_y - 12}" x2="{x_thresh:.1f}" y2="{bar_y + bar_height + 12}" 
              stroke="#34352F" stroke-width="2.5" stroke-dasharray="4 2" />
        
        <!-- Threshold Top Badge -->
        <g transform="translate({x_thresh:.1f}, {thresh_badge_y})">
            <rect x="-65" y="-18" width="130" height="24" rx="4" fill="#34352F" filter="url(#shadow)" />
            <text x="0" y="-2" font-family="Inter, system-ui, sans-serif" font-size="11" font-weight="600" 
                  fill="#FFFCF6" text-anchor="middle" font-variant-numeric="tabular-nums">
                Threshold &tau;: {threshold_f:+.4f}
            </text>
            <polygon points="-5,6 5,6 0,11" fill="#34352F" />
        </g>

        <!-- Email Decision Margin Marker (Pin & Score) -->
        <g transform="translate({x_margin:.1f}, {bar_y + bar_height / 2})">
            <!-- Outer Glow / Ring -->
            <circle cx="0" cy="0" r="14" fill="{status_color}" fill-opacity="0.25" />
            <!-- Inner Solid Node -->
            <circle cx="0" cy="0" r="8.5" fill="{status_color}" stroke="#FFFCF6" stroke-width="2.5" filter="url(#shadow)" />
        </g>

        <!-- Margin Bottom Badge -->
        <g transform="translate({x_margin:.1f}, {margin_badge_y})">
            <polygon points="-5,-11 5,-11 0,-6" fill="{status_color}" />
            <rect x="-70" y="-6" width="140" height="26" rx="4" fill="{status_color}" filter="url(#shadow)" />
            <text x="0" y="11" font-family="Inter, system-ui, sans-serif" font-size="11.5" font-weight="700" 
                  fill="#FFFCF6" text-anchor="middle" font-variant-numeric="tabular-nums">
                Margin: {margin_f:+.4f}
            </text>
        </g>

        <!-- Boundary Delta Arrow / Line if separated -->
        {"<!-- Delta connector -->" if distance < 0.05 else f'''
        <line x1="{x_thresh:.1f}" y1="{bar_y + bar_height + 3}" x2="{x_margin:.1f}" y2="{bar_y + bar_height + 3}" 
              stroke="{status_color}" stroke-width="2" stroke-linecap="round" />
        '''}
    </svg>

    <!-- Sub-Gauge Diagnostic Strip -->
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #F0EDE5;
        border: 1px solid #E4DED2;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.825rem;
    ">
        <div>
            <span style="color: #6F7067; font-weight: 500;">Diagnostic Verdict:</span>
            <strong style="color: {status_color}; margin-left: 0.35rem;">{status_label}</strong>
            <span style="color: #89897F; margin: 0 0.5rem;">|</span>
            <span style="color: #34352F;">{zone_class_label}</span>
        </div>
        <div style="font-variant-numeric: tabular-nums;">
            <span style="color: #6F7067; font-weight: 500;">Boundary Delta (&Delta; = Margin - &tau;):</span>
            <strong style="color: {status_color}; margin-left: 0.35rem;">{delta:+.4f}</strong>
            <span style="color: #6F7067; margin-left: 0.25rem;">({distance:.4f} units from threshold)</span>
        </div>
    </div>
</div>
"""
    return "\n".join(line.strip() for line in svg.splitlines() if line.strip())


def render_margin_gauge(
    margin: float,
    threshold: float,
    model_name: str = "LinearSVC",
) -> None:
    """Render the decision boundary margin gauge inside a Streamlit application.

    Gracefully renders via st.markdown(..., unsafe_allow_html=True).
    """
    html_markup = generate_margin_gauge_svg(margin, threshold, model_name)
    st.markdown(html_markup, unsafe_allow_html=True)

