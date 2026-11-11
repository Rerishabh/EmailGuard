"""Confusion Matrices & Final Held-Out Evaluation component for EmailGuard.

Presents the final held-out test evaluation (N = 641) on the untouched test partition,
annotated confusion matrix heatmap, exact Precision-Recall curve plotted from actual test predictions,
False Positive Rate analysis, and evaluation interpretations.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import pandas as pd


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame:
    """Load the Stage 6D test predictions CSV containing all 641 test records."""
    import pandas as pd

    root = Path(__file__).resolve().parent.parent.parent
    pred_path = root / "results" / "stage6d_test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Test predictions CSV not found at {pred_path}")
    return pd.read_csv(pred_path)


def render_confusion_matrix_heatmap() -> plt.Figure:
    """Render an annotated confusion matrix heatmap for Pipeline B (Held-Out Test N = 641)."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=100)
    fig.patch.set_facecolor("#FFFCF6")
    ax.set_facecolor("#FFFCF6")

    cm = np.array([[543, 2], [4, 92]])
    labels = [
        ["True Negative (TN)\n543 (84.71%)\nCorrectly Allowed", "False Positive (FP)\n2 (0.31%)\nFalse Alarm (Blocked)"],
        ["False Negative (FN)\n4 (0.62%)\nMissed Threat", "True Positive (TP)\n92 (14.35%)\nSuccessfully Intercepted"],
    ]

    # Custom color matrix: Warm Minimalist palette
    cell_colors = [
        ["#EAF1E8", "#F7E9E5"],  # TN (soft sage), FP (soft coral alert)
        ["#FBF3D5", "#EAF1E8"],  # FN (soft warm amber), TP (soft sage)
    ]

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            txt = labels[i][j]
            bg = cell_colors[i][j]
            rect = plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, facecolor=bg, edgecolor="#E4DED2", linewidth=1.5)
            ax.add_patch(rect)
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#34352F", linespacing=1.4)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Legitimate (Ham)", "Predicted Malicious (Spam)"], fontsize=8.5, fontweight="bold", color="#34352F")
    ax.set_yticklabels(["Actual Legitimate\n(Ham: 545)", "Actual Malicious\n(Spam: 96)"], fontsize=8.5, fontweight="bold", color="#34352F")
    ax.set_xlabel("Predicted Security Classification", fontsize=9, fontweight="bold", color="#6F7067", labelpad=8)
    ax.set_ylabel("Ground Truth Annotation", fontsize=9, fontweight="bold", color="#6F7067", labelpad=8)
    ax.set_title("Champion Confusion Matrix (Held-Out Test N = 641)", fontsize=10.5, fontweight="bold", color="#34352F", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.tight_layout()
    return fig


def render_precision_recall_curve_plot(df_preds: pd.DataFrame) -> plt.Figure:
    """Plot the actual Precision-Recall curve using 641 test records for Pipeline B."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    df_b = df_preds[df_preds["candidate_name"] == "pipeline_b_linear_svc"]
    y_true = df_b["y_true"].values
    scores = df_b["continuous_score"].values

    precisions, recalls, _ = precision_recall_curve(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=100)
    fig.patch.set_facecolor("#FFFCF6")
    ax.set_facecolor("#FFFCF6")

    # Plot PR Curve with Warm Minimalist accent
    ax.plot(recalls, precisions, color="#718C83", linewidth=2.5, label="Pipeline B (PR-AUC = 0.9978)")
    ax.fill_between(recalls, precisions, alpha=0.15, color="#718C83")

    # Mark the calibrated operating point (tau = 0.004085)
    cal_rec = 92.0 / 96.0
    cal_prec = 92.0 / 94.0
    ax.scatter([cal_rec], [cal_prec], color="#64866C", s=130, zorder=6, edgecolors="#34352F", linewidth=1.5,
               label=f"Calibrated Threshold Operating Point (P={cal_prec:.3f}, R={cal_rec:.3f})")

    # Mark baseline no-skill line (proportion of positives = 96 / 641 = 0.1498)
    ax.axhline(y=96 / 641, color="#C47770", linestyle=":", linewidth=1.2, label=f"No-Skill Baseline ({96/641:.2%})")

    ax.set_xlabel("Recall (Fraction of True Spam Intercepted)", fontsize=8.5, fontweight="bold", color="#6F7067")
    ax.set_ylabel("Precision (Fraction of Flagged Emails that are Spam)", fontsize=8.5, fontweight="bold", color="#6F7067")
    ax.set_title("Empirical Precision-Recall Curve (Stage 6D Test Set)", fontsize=10.5, fontweight="bold", color="#34352F", pad=12)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=7.5, loc="lower left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E4DED2")
    ax.spines["bottom"].set_color("#E4DED2")
    ax.tick_params(colors="#6F7067", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.5, color="#E4DED2")
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def _figure_to_png_bytes(fig: plt.Figure) -> bytes:
    """Serialize a Matplotlib figure to PNG bytes and close it to prevent memory leaks."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
        return buf.getvalue()
    finally:
        plt.close(fig)


@st.cache_data(show_spinner=False)
def _get_confusion_matrix_png() -> bytes:
    """Render and cache the Confusion Matrix heatmap PNG."""
    return _figure_to_png_bytes(render_confusion_matrix_heatmap())


@st.cache_data(show_spinner=False)
def _get_pr_curve_png() -> bytes:
    """Render and cache the Empirical Precision-Recall Curve PNG."""
    df_preds = load_test_predictions()
    return _figure_to_png_bytes(render_precision_recall_curve_plot(df_preds))


def render_evaluation_tab() -> None:
    """Render the complete Confusion Matrices & Final Evaluation tab."""
    st.markdown(
        """
        <div style="
            background: #FFFCF6; border: 1px solid #E4DED2; border-radius: 8px;
            padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; box-shadow: 0 1px 3px rgba(52,53,47,0.04);
        ">
            <h3 style="margin: 0 0 0.25rem 0; font-size: 1.25rem; color: #34352F; font-weight: 700;">
                Stage 6D: Final Held-Out Test Evaluation (N = 641)
            </h3>
            <p style="margin: 0; font-size: 0.875rem; color: #6F7067; line-height: 1.5;">
                Following all tuning and threshold calibrations on the training partition, the frozen production champion
                (<strong>Pipeline B + LinearSVC</strong>) was evaluated on the untouched 641-record held-out test partition.
                These metrics represent true out-of-sample operational generalization.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section B: Final Performance Metrics Cards
    st.markdown(
        """
        <div style="font-size: 0.75rem; font-weight: 700; color: #89897F; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
            Verified Held-Out Performance Metrics (Untouched Test Partition)
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Accuracy", "99.064%", "543 TN + 92 TP / 641")
    m2.metric("Spam Precision", "97.872%", "92 TP / (92 TP + 2 FP)")
    m3.metric("Spam Recall", "95.833%", "92 TP / (92 TP + 4 FN)")
    m4.metric("Spam F1-Score", "96.842%", "Harmonic Mean of P & R")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Not Spam Precision", "99.269%", "543 TN / (543 TN + 4 FN)")
    m6.metric("Not Spam Recall", "99.633%", "543 TN / (543 TN + 2 FP)")
    m7.metric("False Positive Rate", "0.367%", "2 FP / 545 Legitimate")
    m8.metric("Average Precision (PR-AUC)", "0.997796", "Exact 640-Point PR Area")

    # Section A & Section D: Confusion Matrix & PR Curve
    col_cm, col_pr = st.columns(2)

    with col_cm:
        st.markdown(
            """
            <div style="
                background: #FFFCF6; border: 1px solid #E4DED2; border-radius: 8px;
                padding: 1rem 1.25rem; height: 100%; box-shadow: 0 1px 2px rgba(52,53,47,0.03);
            ">
            """,
            unsafe_allow_html=True,
        )
        st.image(_get_confusion_matrix_png(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_pr:
        st.markdown(
            """
            <div style="
                background: #FFFCF6; border: 1px solid #E4DED2; border-radius: 8px;
                padding: 1rem 1.25rem; height: 100%; box-shadow: 0 1px 2px rgba(52,53,47,0.03);
            ">
            """,
            unsafe_allow_html=True,
        )
        st.image(_get_pr_curve_png(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Section C & Section E: False Positive Rate Analysis & Academic Interpretation
    st.markdown(
        """
        <div style="
            background: #FFFCF6; border: 1px solid #E4DED2; border-radius: 8px;
            padding: 1.25rem 1.5rem; margin-top: 1.25rem; box-shadow: 0 1px 2px rgba(52,53,47,0.03);
        ">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 1.05rem; color: #34352F; font-weight: 700;">
                False Positive Rate (FPR) Protocol & Enterprise Risk Profile
            </h4>
            <div style="font-size: 0.85rem; color: #34352F; line-height: 1.6;">
                <p>
                    <strong>What is a False Positive in Email Security?</strong><br>
                    A False Positive (FP) occurs when a legitimate benign email (such as a mission-critical client communication, contract, or password reset) is mistakenly categorized as malicious spam and dropped or quarantined.
                    In commercial cybersecurity, <em>False Positives carry a vastly higher operational cost than occasional False Negatives</em>.
                </p>
                <p>
                    <strong>Why the &le; 0.50% FPR Threshold Constraint Was Enforced:</strong><br>
                    During Stage 6C cross-validation, the operating threshold was calibrated specifically to satisfy an upper bound of 0.50% FPR on the training distribution.
                    On the unseen held-out test partition, the champion achieved an <strong>FPR of 0.367% (only 2 false alarms out of 545 benign emails)</strong>, confirming that the calibration successfully generalized without overfitting.
                    <em>We do not claim zero false positives</em>, as statistical text classification always operates under non-zero noise.
                </p>
            </div>
        </div>

        <div style="
            background: #FFFCF6; border: 1px solid #E4DED2; border-radius: 8px;
            padding: 1.25rem 1.5rem; margin-top: 1.25rem; box-shadow: 0 1px 2px rgba(52,53,47,0.03);
        ">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 1.05rem; color: #34352F; font-weight: 700;">
                Academic Evaluation & Methodological Insights
            </h4>
            <div style="font-size: 0.85rem; color: #34352F; line-height: 1.6;">
                <p>
                    <strong>1. Why Accuracy Alone is Deceptive:</strong><br>
                    Because 84.97% of the corpus consists of legitimate emails, a naive "dummy" classifier that automatically labels every message as "Ham" achieves an impressive 84.97% accuracy while detecting exactly 0% of incoming cyber threats.
                    Therefore, <strong>Spam Precision</strong>, <strong>Spam Recall</strong>, and <strong>Average Precision (PR-AUC)</strong> serve as our primary evaluation criteria.
                </p>
                <p>
                    <strong>2. Precision vs. Recall Dynamics:</strong><br>
                    With 97.87% Precision, only ~2.1% of emails flagged by EmailGuard are misclassified.
                    With 95.83% Recall, 92 out of 96 active spam messages were neutralized, yielding an <strong>F1-score of 96.84%</strong>.
                </p>
                <p>
                    <strong>3. Held-Out Generalization Integrity:</strong><br>
                    Cross-validation F1 in Stage 6C was 92.95%, whereas held-out test F1 in Stage 6D reached 96.84%.
                    This confirmed that the model learned robust, non-spurious language representations that generalized effectively to novel messages.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

