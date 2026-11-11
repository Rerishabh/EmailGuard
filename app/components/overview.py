"""Overview & Dataset Summary component for EmailGuard Streamlit Application.

Presents a compact dataset distribution breakdown (2,720 Not Spam vs. 481 Spam)
and a clean visual distribution chart without repetitive paragraphs.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def render_corpus_distribution_chart() -> plt.Figure:
    """Generate an enterprise-styled compact class distribution bar chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 1.8), dpi=100)
    fig.patch.set_facecolor("#FFFCF6")
    ax.set_facecolor("#FFFCF6")

    categories = ["Not Spam (Ham)", "Spam"]
    counts = [2720, 481]
    percentages = [84.97, 15.03]
    colors = ["#64866C", "#C47770"]

    bars = ax.barh(categories, counts, color=colors, height=0.42, edgecolor="#E4DED2")

    # Add count and percentage labels
    for bar, count, pct in zip(bars, counts, percentages):
        w = bar.get_width()
        ax.text(
            w + 40,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} emails ({pct:.2f}%)",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#0F172A",
        )

    ax.set_xlim(0, 3300)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="y", labelsize=8.5, colors="#0F172A")
    ax.tick_params(axis="x", labelsize=8, colors="#64748B")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, color="#CBD5E1")
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def _get_corpus_chart_png() -> bytes:
    """Render and cache the corpus distribution figure as PNG bytes to avoid per-rerun redraws."""
    import matplotlib.pyplot as plt

    fig = render_corpus_distribution_chart()
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
        return buf.getvalue()
    finally:
        plt.close(fig)


def render_dataset_summary() -> None:
    """Render the compact dataset distribution summary below the live analyzer."""
    with st.container(border=True):
        st.subheader("Corpus Dataset Distribution Summary")
        col_metrics, col_chart = st.columns([1, 1])

        with col_metrics:
            st.write(
                "The benchmark corpus consists of **3,201 total cleaned emails** from the SpamAssassin Public Corpus, "
                "partitioned into 2,560 training records and 641 held-out test records:"
            )
            c1, c2 = st.columns(2)
            c1.metric("Not Spam (Ham)", "2,720", "84.97% of Corpus")
            c2.metric("Spam (Malicious)", "481", "15.03% of Corpus")

        with col_chart:
            st.image(_get_corpus_chart_png(), use_container_width=True)


def render_project_overview() -> None:
    """Backward-compatible entry point for dataset summary."""
    render_dataset_summary()
