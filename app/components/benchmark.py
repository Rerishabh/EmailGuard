"""6-Model Benchmark component for EmailGuard Streamlit Application.

Presents academic cross-validation comparisons across all six candidate classifiers,
interactive Matplotlib/Seaborn performance visualizations, research findings,
and hyperparameter tuning specifications from Stage 6B and Stage 6C.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import streamlit as st

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import pandas as pd


@st.cache_data(show_spinner=False)
def load_cv_summary_data() -> pd.DataFrame:
    """Load the Stage 6B 5-fold cross-validation summary CSV."""
    import pandas as pd

    root = Path(__file__).resolve().parent.parent.parent
    csv_path = root / "results" / "stage6b_cv_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cross-validation summary CSV not found at {csv_path}")
    return pd.read_csv(csv_path)


def format_classifier_name(raw_name: str) -> str:
    """Map internal classifier key to readable publication name."""
    mapping = {
        "linear_svc": "LinearSVC (Support Vector Classifier)",
        "logistic_regression": "Logistic Regression",
        "multinomial_nb": "Multinomial Naive Bayes",
        "random_forest": "Random Forest",
        "decision_tree": "Decision Tree",
        "knn": "K-Nearest Neighbors (KNN)",
    }
    return mapping.get(raw_name, raw_name.replace("_", " ").title())


def _build_benchmark_figures(
    df_pipe: pd.DataFrame, pipe_label: str
) -> Tuple[plt.Figure, plt.Figure, plt.Figure, plt.Figure]:
    """Create the 4 comparative Matplotlib figures for the selected pipeline."""
    import matplotlib.pyplot as plt

    # Ensure ordered by F1-score
    df_sorted = df_pipe.sort_values(by="f1_spam_mean", ascending=True).copy()
    model_labels = [format_classifier_name(c).split(" (")[0] for c in df_sorted["classifier"]]

    # Chart 1: Spam F1
    fig1, ax1 = plt.subplots(figsize=(6, 3.2), dpi=100)
    fig1.patch.set_facecolor("#FFFCF6")
    ax1.set_facecolor("#FFFCF6")

    colors1 = ["#718C83" if "linear_svc" in c else "#89897F" for c in df_sorted["classifier"]]
    bars1 = ax1.barh(
        model_labels,
        df_sorted["f1_spam_mean"],
        xerr=df_sorted["f1_spam_std"],
        color=colors1,
        height=0.5,
        capsize=3,
        edgecolor="#E4DED2",
    )

    for bar, val, std in zip(bars1, df_sorted["f1_spam_mean"], df_sorted["f1_spam_std"]):
        ax1.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f} ± {std:.3f}",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
            color="#34352F",
        )

    ax1.set_xlim(0, 1.15)
    ax1.set_title(f"Spam F1-Score (5-Fold CV: {pipe_label})", fontsize=10, fontweight="bold", color="#34352F", pad=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#E4DED2")
    ax1.spines["bottom"].set_color("#E4DED2")
    ax1.tick_params(axis="y", labelsize=8, colors="#34352F")
    ax1.tick_params(axis="x", labelsize=8, colors="#6F7067")
    ax1.xaxis.grid(True, linestyle="--", alpha=0.5, color="#E4DED2")
    ax1.set_axisbelow(True)
    plt.tight_layout()

    # Chart 2: FPR
    fig2, ax2 = plt.subplots(figsize=(6, 3.2), dpi=100)
    fig2.patch.set_facecolor("#FFFCF6")
    ax2.set_facecolor("#FFFCF6")

    # Color based on whether FPR satisfies <= 0.005 threshold
    colors2 = ["#64866C" if val <= 0.005 else "#C47770" for val in df_sorted["fpr_mean"]]
    bars2 = ax2.barh(
        model_labels,
        df_sorted["fpr_mean"] * 100,
        xerr=df_sorted["fpr_std"] * 100,
        color=colors2,
        height=0.5,
        capsize=3,
        edgecolor="#E4DED2",
    )

    for bar, val in zip(bars2, df_sorted["fpr_mean"] * 100):
        ax2.text(
            val + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
            color="#34352F",
        )

    # Tolerance threshold line at 0.50%
    ax2.axvline(x=0.5, color="#C47770", linestyle="--", linewidth=1.5, label="Max FPR Tolerance (0.50%)")
    ax2.set_title("False Positive Rate % (Lower is Better)", fontsize=10, fontweight="bold", color="#34352F", pad=10)
    ax2.legend(fontsize=7, loc="lower right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#E4DED2")
    ax2.spines["bottom"].set_color("#E4DED2")
    ax2.tick_params(axis="y", labelsize=8, colors="#34352F")
    ax2.tick_params(axis="x", labelsize=8, colors="#6F7067")
    ax2.xaxis.grid(True, linestyle="--", alpha=0.5, color="#E4DED2")
    ax2.set_axisbelow(True)
    plt.tight_layout()

    # Chart 3: Average Precision / PR-AUC
    fig3, ax3 = plt.subplots(figsize=(6, 3.2), dpi=100)
    fig3.patch.set_facecolor("#FFFCF6")
    ax3.set_facecolor("#FFFCF6")

    colors3 = ["#718C83" if "linear_svc" in c else "#89897F" for c in df_sorted["classifier"]]
    bars3 = ax3.barh(
        model_labels,
        df_sorted["pr_auc_average_precision_mean"],
        xerr=df_sorted["pr_auc_average_precision_std"],
        color=colors3,
        height=0.5,
        capsize=3,
        edgecolor="#E4DED2",
    )

    for bar, val in zip(bars3, df_sorted["pr_auc_average_precision_mean"]):
        ax3.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
            color="#34352F",
        )

    ax3.set_xlim(0.5, 1.15)
    ax3.set_title("Average Precision / PR-AUC (Ranking Quality)", fontsize=10, fontweight="bold", color="#34352F", pad=10)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_color("#E4DED2")
    ax3.spines["bottom"].set_color("#E4DED2")
    ax3.tick_params(axis="y", labelsize=8, colors="#34352F")
    ax3.tick_params(axis="x", labelsize=8, colors="#6F7067")
    ax3.xaxis.grid(True, linestyle="--", alpha=0.5, color="#E4DED2")
    ax3.set_axisbelow(True)
    plt.tight_layout()

    # Chart 4: Precision vs Recall Scatter
    fig4, ax4 = plt.subplots(figsize=(6, 3.2), dpi=100)
    fig4.patch.set_facecolor("#FFFCF6")
    ax4.set_facecolor("#FFFCF6")

    for _, row in df_pipe.iterrows():
        c_key = row["classifier"]
        prec = row["precision_spam_mean"]
        rec = row["recall_spam_mean"]
        is_champ = c_key == "linear_svc"
        color = "#718C83" if is_champ else "#89897F"
        size = 120 if is_champ else 60
        name = format_classifier_name(c_key).split(" (")[0]

        ax4.scatter(rec, prec, color=color, s=size, edgecolors="#34352F", zorder=5)
        offset_y = 0.015 if prec > 0.9 else -0.025
        ax4.annotate(
            name,
            (rec, prec + offset_y),
            fontsize=7.5,
            fontweight="bold" if is_champ else "normal",
            ha="center",
            color="#34352F",
        )

    ax4.set_xlabel("Spam Recall", fontsize=8, color="#6F7067")
    ax4.set_ylabel("Spam Precision", fontsize=8, color="#6F7067")
    ax4.set_xlim(0.2, 1.05)
    ax4.set_ylim(0.7, 1.05)
    ax4.set_title("Spam Precision vs. Recall Trade-Off", fontsize=10, fontweight="bold", color="#34352F", pad=10)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.spines["left"].set_color("#E4DED2")
    ax4.spines["bottom"].set_color("#E4DED2")
    ax4.tick_params(colors="#6F7067", labelsize=8)
    ax4.grid(True, linestyle="--", alpha=0.5, color="#E4DED2")
    ax4.set_axisbelow(True)
    plt.tight_layout()

    return fig1, fig2, fig3, fig4


def render_benchmark_charts(df_pipe: pd.DataFrame, pipe_label: str) -> None:
    """Render 4 comparative charts for the selected pipeline."""
    import matplotlib.pyplot as plt

    fig1, fig2, fig3, fig4 = _build_benchmark_figures(df_pipe, pipe_label)
    try:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.pyplot(fig1, use_container_width=True)
        with col_c2:
            st.pyplot(fig2, use_container_width=True)

        col_c3, col_c4 = st.columns(2)
        with col_c3:
            st.pyplot(fig3, use_container_width=True)
        with col_c4:
            st.pyplot(fig4, use_container_width=True)
    finally:
        for fig in (fig1, fig2, fig3, fig4):
            plt.close(fig)


@st.cache_data(show_spinner=False)
def _get_benchmark_charts_png(pipe_choice: str, pipe_label: str) -> Tuple[bytes, bytes, bytes, bytes]:
    """Render and cache the 4 benchmark charts as PNG bytes for a pipeline."""
    import matplotlib.pyplot as plt

    cv_df = load_cv_summary_data()
    df_pipe = cv_df[cv_df["pipeline"] == pipe_choice].copy()
    figs = _build_benchmark_figures(df_pipe, pipe_label)
    pngs: List[bytes] = []
    try:
        for fig in figs:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
            pngs.append(buf.getvalue())
        return pngs[0], pngs[1], pngs[2], pngs[3]
    finally:
        for fig in figs:
            plt.close(fig)


@st.cache_data(show_spinner=False)
def _get_benchmark_display_table(pipe_choice: str) -> pd.DataFrame:
    """Build and cache the formatted 6-model benchmark summary DataFrame."""
    import pandas as pd

    cv_df = load_cv_summary_data()
    df_pipe = cv_df[cv_df["pipeline"] == pipe_choice]
    display_rows = []
    for _, row in df_pipe.sort_values(by="f1_spam_mean", ascending=False).iterrows():
        display_rows.append({
            "Classifier": format_classifier_name(row["classifier"]),
            "Accuracy": f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}",
            "Spam Precision": f"{row['precision_spam_mean']:.4f} ± {row['precision_spam_std']:.4f}",
            "Spam Recall": f"{row['recall_spam_mean']:.4f} ± {row['recall_spam_std']:.4f}",
            "Spam F1-Score": f"{row['f1_spam_mean']:.4f} ± {row['f1_spam_std']:.4f}",
            "False Positive Rate": f"{row['fpr_mean']:.4f} ± {row['fpr_std']:.4f}",
            "Average Precision": f"{row['pr_auc_average_precision_mean']:.4f} ± {row['pr_auc_average_precision_std']:.4f}",
            "Mean Train Time (s)": f"{row['t_train_s_mean']:.2f}s",
            "Mean Infer Time (s)": f"{row['t_infer_s_mean']:.2f}s",
        })
    return pd.DataFrame(display_rows)


def render_benchmark_tab() -> None:
    """Render the complete 6-Model Benchmark tab."""
    st.markdown(
        """
        <div style="
            background: #FFFCF6; border: 1px solid #E4DED2; border-radius: 8px;
            padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; box-shadow: 0 1px 3px rgba(52,53,47,0.04);
        ">
            <h3 style="margin: 0 0 0.25rem 0; font-size: 1.25rem; color: #34352F; font-weight: 700;">
                Academic Evaluation: 6-Model Cross-Validation Benchmark
            </h3>
            <p style="margin: 0; font-size: 0.875rem; color: #6F7067; line-height: 1.5;">
                To establish an empirical baseline, six distinct machine learning algorithms were benchmarked across identical stratified 5-fold cross-validation splits on the 2,560-sample training partition.
                All metrics reflect out-of-fold generalization prior to final held-out test verification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Pipeline selection
    col_sel, _ = st.columns([2, 3])
    with col_sel:
        pipe_choice = st.selectbox(
            "Select Feature Engineering Pipeline:",
            options=["pipeline_b", "pipeline_c", "pipeline_a"],
            format_func=lambda p: {
                "pipeline_b": "Pipeline B: Standard NLP (Production Champion)",
                "pipeline_c": "Pipeline C: Domain-Aware + Scaled Heuristics",
                "pipeline_a": "Pipeline A: Minimal Preprocessing",
            }[p],
            index=0,
        )

    pipe_label = {"pipeline_b": "Pipeline B", "pipeline_c": "Pipeline C", "pipeline_a": "Pipeline A"}[pipe_choice]

    # Section A: Results Table
    st.markdown(
        f"""
        <h4 style="margin: 1.25rem 0 0.5rem 0; font-size: 1.05rem; color: #34352F; font-weight: 700;">
            Comparative Cross-Validation Performance ({pipe_label})
        </h4>
        <p style="font-size: 0.8rem; color: #6F7067; margin-bottom: 0.75rem;">
            Source: <code>results/stage6b_cv_summary.csv</code> | 5-Fold Stratified Cross-Validation (Mean &plusmn; Std Dev)
        </p>
        """,
        unsafe_allow_html=True,
    )

    table_df = _get_benchmark_display_table(pipe_choice)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # Section B: Comparative Visualizations
    st.markdown(
        """
        <h4 style="margin: 1.5rem 0 0.75rem 0; font-size: 1.05rem; color: #34352F; font-weight: 700;">
            Benchmark Diagnostic Visualizations
        </h4>
        """,
        unsafe_allow_html=True,
    )
    png1, png2, png3, png4 = _get_benchmark_charts_png(pipe_choice, pipe_label)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.image(png1, use_container_width=True)
    with col_c2:
        st.image(png2, use_container_width=True)
    col_c3, col_c4 = st.columns(2)
    with col_c3:
        st.image(png3, use_container_width=True)
    with col_c4:
        st.image(png4, use_container_width=True)

    # Section C: Research Findings & Empirical Analysis
    st.markdown(
        """
        <div style="
            background: #FFFCF6; border: 1px solid #E4DED2; border-radius: 8px;
            padding: 1.25rem 1.5rem; margin-top: 1.25rem; box-shadow: 0 1px 2px rgba(52,53,47,0.03);
        ">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 1.05rem; color: #34352F; font-weight: 700;">
                Empirical Research Findings & Algorithmic Rationale
            </h4>
            <div style="font-size: 0.85rem; color: #34352F; line-height: 1.6;">
                <p>
                    <strong>1. Why Linear Classifiers Excel on Sparse TF-IDF:</strong><br>
                    Text representation generates a high-dimensional sparse feature space (&gt;20,000 unigram and bigram tokens).
                    In such high-dimensional spaces, text corpora are almost always linearly separable.
                    <strong>LinearSVC</strong> identifies the maximum-margin hyperplane separating classes, maximizing geometric generalization and resisting overfitting far better than non-linear tree ensembles.
                </p>
                <p>
                    <strong>2. Champion Selection Protocol:</strong><br>
                    Under the pre-declared operational constraint (False Positive Rate &le; 0.50% and maximum F1), <strong>Pipeline B + LinearSVC</strong> demonstrated optimal stability:
                    achieving <strong>0.9176 F1</strong>, <strong>0.9689 Spam Precision</strong>, and an impressive <strong>0.9860 Average Precision</strong> during cross-validation, while strictly controlling FPR to 0.0051.
                </p>
                <p>
                    <strong>3. The KNN Anomaly:</strong><br>
                    K-Nearest Neighbors achieved high accuracy (0.9727) and F1 (0.9047) in Pipeline B because cosine similarity in normalized TF-IDF space clusters clean ham and spam emails tightly.
                    However, KNN requires <em>O(N)</em> distance computations against all 2,560 training vectors at inference time, rendering its inference latency (&gt;7.6 ms/batch) unacceptable for line-rate mail servers.
                </p>
                <p>
                    <strong>4. Tree Ensemble Limitations:</strong><br>
                    Decision Trees (F1: 0.7609) and Random Forests (F1: 0.7861) suffered from orthogonal axis-aligned splitting limitations across thousands of sparse zero-valued features, yielding suboptimal recall (0.6597 for Random Forest).
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section D: Hyperparameter Tuning Specifications
    st.markdown(
        """
        <div style="
            background: #FFFCF6; border: 1px solid #E4DED2; border-radius: 8px;
            padding: 1.25rem 1.5rem; margin-top: 1.25rem; box-shadow: 0 1px 2px rgba(52,53,47,0.03);
        ">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 1.05rem; color: #34352F; font-weight: 700;">
                Stage 6C Hyperparameter Tuning Specifications
            </h4>
            <p style="font-size: 0.825rem; color: #6F7067; margin-bottom: 0.75rem;">
                Hyperparameters and operating thresholds were tuned via nested training-side cross-validation. The held-out test partition remained untouched.
            </p>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                <div style="background: #F0EDE5; border: 1px solid #E4DED2; border-radius: 6px; padding: 1rem;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #5E786F; margin-bottom: 0.4rem;">
                        Pipeline B + LinearSVC (Production Champion)
                    </div>
                    <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.8rem; color: #34352F; line-height: 1.6;">
                        <li><strong>Regularization (C):</strong> 2.0 (Balanced margin penalty)</li>
                        <li><strong>Loss Function:</strong> Hinge Loss (Traditional SVM)</li>
                        <li><strong>Dual Optimization:</strong> True (Solves dual problem for N &lt; D)</li>
                        <li><strong>Max Iterations:</strong> 5,000</li>
                        <li><strong>Calibrated Threshold (&tau;<sup>*</sup>):</strong> <code>+0.0040852377135018</code></li>
                        <li><strong>FPR Constraint:</strong> Satisfies empirical FPR &le; 0.50%</li>
                    </ul>
                </div>
                <div style="background: #F0EDE5; border: 1px solid #E4DED2; border-radius: 6px; padding: 1rem;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #34352F; margin-bottom: 0.4rem;">
                        Pipeline C + LinearSVC (High-Throughput Alternative)
                    </div>
                    <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.8rem; color: #34352F; line-height: 1.6;">
                        <li><strong>Regularization (C):</strong> 10.0</li>
                        <li><strong>Loss Function:</strong> Squared Hinge Loss (Smooth gradient)</li>
                        <li><strong>Dual Optimization:</strong> True</li>
                        <li><strong>Heuristic Weight:</strong> 0.05 (StandardScaler scaled)</li>
                        <li><strong>Calibrated Threshold (&tau;<sup>*</sup>):</strong> <code>+0.0876861391187918</code></li>
                        <li><strong>Latency Advantage:</strong> 2.72x faster inference (2.02 ms/email)</li>
                    </ul>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

