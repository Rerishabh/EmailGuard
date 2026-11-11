"""Architecture & Methodology component for EmailGuard Streamlit Application.

Presents complete system architecture, ML pipeline workflows, comparative feature engineering
topologies (Pipelines A, B, C), hyperparameter configurations, and latency benchmarks.
"""

from __future__ import annotations

import streamlit as st


def render_methodology_tab() -> None:
    """Render the complete Architecture & Methodology tab."""
    with st.container(border=True):
        st.subheader("System Architecture & Machine Learning Methodology")
        st.write(
            "EmailGuard implements an end-to-end, reproducible NLP security workflow. "
            "From raw multi-part email parsing and regex extraction to sparse TF-IDF vectorization and support vector classification, "
            "every component is strictly compartmentalized to prevent data leakage and ensure line-rate operational throughput."
        )

    # Section A: Dataset & Historical Scope
    with st.container(border=True):
        st.subheader("A. Benchmark Dataset & Corpus Scope")
        st.write(
            "**Corpus Source:** The benchmark is constructed from the widely recognized **SpamAssassin Public Corpus** (~2002/2003). "
            "Following rigorous email parsing and byte deduplication, the cleaned dataset consists of exactly **3,201 unique emails**:"
        )
        c1, c2 = st.columns(2)
        c1.info("**Legitimate Emails (Ham):** 2,720 records (84.97%)")
        c2.warning("**Malicious Emails (Spam):** 481 records (15.03%)")

        st.caption(
            "⚠️ **Methodological Notice on Historical Data:** While the SpamAssassin corpus represents an academic standard for reproducible "
            "NLP filtering research, it reflects early-2000s threat patterns. Modern production email gateways encounter adversarial evasion "
            "techniques (e.g., zero-day URL obfuscation, homoglyphs, image steganography, and LLM-synthesized phishing) "
            "which necessitate continuous adaptive retraining and multi-layered defense-in-depth beyond static text classification."
        )

    # Section B: ML Workflow Flowchart
    with st.container(border=True):
        st.subheader("B. End-to-End Machine Learning Pipeline Workflow")
        st.markdown(
            "The project executes across a strictly partitioned 12-stage pipeline to ensure zero data leakage between training and testing:"
        )

        flowchart_steps = [
            "**Phase 1: Ingestion & Preparation** — 1. Corpus Ingestion → 2. BytesParser Email Extraction → 3. Full-Text Deduplication (3,201) → 4. Stratified Train/Test Split (80/20)",
            "**Phase 2: Modeling & Validation** — 5. NLP Normalization → 6. TF-IDF (1-2 N-Grams) → 7. 6-Model 5-Fold Stratified CV → 8. Hyperparameter & Threshold Calibration",
            "**Phase 3: Production & Verification** — 9. Champion Selection (Pipeline B) → 10. Held-Out Evaluation (N=641) → 11. Pipeline Serialization (.joblib) → 12. Live Streamlit Deployment",
        ]
        for step in flowchart_steps:
            st.markdown(f"- {step}")

    # Section C: Three Feature Pipelines (Native Streamlit Columns & Containers)
    st.subheader("C. Comparative Pipeline Architectures")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        with st.container(border=True):
            st.caption("PIPELINE A: MINIMAL NLP")
            st.markdown(
                "**Topology:** MinimalTextTransformer → TF-IDF → Classifier"
            )
            st.write(
                "Basic case folding and whitespace collapsing."
            )

    with col_b:
        with st.container(border=True):
            st.caption("PIPELINE B: STANDARD NLP (CHAMPION)")
            st.markdown(
                "**Topology:** StandardNLPTextTransformer → TF-IDF → LinearSVC"
            )
            st.write(
                "Linguistic normalization: lowercasing, regex symbol "
                "stripping, stopword removal, and Porter stemming."
            )

    with col_c:
        with st.container(border=True):
            st.caption("PIPELINE C: DOMAIN + HEURISTICS")
            st.markdown(
                "**Topology:** DomainAwareTextTransformer → TF-IDF + "
                "6 Heuristics → Scaling → LinearSVC"
            )
            st.write(
                "Domain-aware text processing and structural email features. "
                "Demonstrated 2.72x faster inference than Pipeline B."
            )

    # Section D & Section E: Pipeline B Champion Config & 6 Heuristic Features
    col_d, col_e = st.columns(2)

    with col_d:
        with st.container(border=True):
            st.subheader("D. Pipeline B Champion Specification")
            st.markdown(
                """
                - **TF-IDF N-Gram Range:** `(1, 2)` (Unigrams & Bigrams)
                - **Min Document Frequency:** `3`
                - **Sublinear TF Scaling:** `Enabled (1 + log(tf))`
                - **Vector Norm:** `L2 normalization`
                - **Classifier:** `LinearSVC (C=2.0, loss='hinge', dual=True, max_iter=5000)`
                - **Calibrated Operating Threshold (&tau;<sup>*</sup>):** `+0.0040852377135018`
                """,
                unsafe_allow_html=True,
            )

    with col_e:
        with st.container(border=True):
            st.subheader("E. Pipeline C Dense Heuristic Features")
            st.markdown(
                """
                Extracted via `src/preprocessing.py`:
                1. **Uppercase Ratio:** Ratio of uppercase letters to total alphabetic letters.
                2. **All-Caps Word Count:** Total words containing &ge;3 consecutive capital letters.
                3. **Currency Symbol Count:** Total count of monetary symbols ($, €, £, ¥).
                4. **URL Count:** Regex count of explicit web URLs and domain links.
                5. **Exclamation Count:** Total count of '!' punctuation marks.
                6. **Question Count:** Total count of '?' punctuation marks.
                """,
                unsafe_allow_html=True,
            )

    # Section F: Latency Comparison
    with st.container(border=True):
        st.subheader("F. Computational Latency & Line-Rate Throughput Benchmark")
        col_l1, col_l2 = st.columns(2)

        with col_l1:
            st.metric("Pipeline B (Champion)", "5.48 ms/email", "Maximum Precision & Generalization")
            st.caption("Applies Porter stemming and stopword filtering. Selected under predefined protocol.")

        with col_l2:
            st.metric("Pipeline C (Alternative)", "2.02 ms/email", "2.72x Faster Inference")
            st.caption("Replaces stemming with fast regex token replacements and 6 heuristics for line-rate MTA filtering.")

        st.caption("* Benchmarks measured on 641 test records under single-thread CPU execution.")
