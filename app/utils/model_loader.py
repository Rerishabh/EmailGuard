"""Model and configuration loader for the Email Spam Detection Streamlit application.

Provides cached loading of the serialized production champion (Pipeline B + LinearSVC)
and cached in-memory fitting of the high-throughput alternative (Pipeline C + LinearSVC)
trained strictly on the verified 2,560-sample training partition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import streamlit as st

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


# Canonical relative paths
CHAMPION_MODEL_PATH = Path("models") / "champion_pipeline.joblib"
CHAMPION_CONFIG_PATH = Path("models") / "champion_config.json"
TRAIN_INDICES_PATH = Path("dataset") / "train_split_indices.npy"
CLEANED_DATASET_PATH = Path("dataset") / "spam_assassin_cleaned.csv"

# Canonical thresholds established in Stage 6C and verified in Stage 6D
PIPELINE_B_DEFAULT_THRESHOLD: float = 0.0
PIPELINE_B_CALIBRATED_THRESHOLD: float = 0.0040852377135018

PIPELINE_C_DEFAULT_THRESHOLD: float = 0.0
PIPELINE_C_CALIBRATED_THRESHOLD: float = 0.0876861391187918

EXPECTED_TRAIN_RECORD_COUNT: int = 2560


def get_project_root() -> Path:
    """Return the absolute path to the project repository root."""
    return Path(__file__).resolve().parent.parent.parent


@st.cache_data
def load_champion_config() -> Dict[str, Any]:
    """Load and return the serialized champion configuration JSON.

    Cached via Streamlit data cache for zero-overhead re-reads.
    """
    root = get_project_root()
    config_file = root / CHAMPION_CONFIG_PATH
    if not config_file.exists():
        raise FileNotFoundError(f"Champion configuration not found at {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)
    return config


@st.cache_resource
def load_champion_model() -> Pipeline:
    """Load the serialized Production Champion (Pipeline B + LinearSVC).

    Loads the frozen joblib artifact verified in Stage 6D.
    Cached across all user sessions using Streamlit resource cache.
    """
    root = get_project_root()
    model_file = root / CHAMPION_MODEL_PATH
    if not model_file.exists():
        raise FileNotFoundError(f"Champion model artifact not found at {model_file}")

    import joblib
    from sklearn.pipeline import Pipeline

    model = joblib.load(model_file)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected sklearn.pipeline.Pipeline, got {type(model)}")

    return model


@st.cache_resource
def load_pipeline_c_model() -> Pipeline:
    """Fit and return the High-Throughput Alternative (Pipeline C + LinearSVC).

    Constructs Pipeline C with Stage 6C optimal hyperparameters:
    - Classifier: LinearSVC(C=10.0, loss='squared_hinge', dual='auto', max_iter=5000)
    - Heuristic weight: 0.05
    - Architecture: DomainAwareTextTransformer + TF-IDF combined with
                    StandardScaler-normalized EmailHeuristicFeatures.

    Strictly fits on the 2,560 training records only. The held-out test split
    is never accessed or referenced. Cached across sessions via Streamlit resource cache.
    """
    root = get_project_root()
    dataset_file = root / CLEANED_DATASET_PATH
    train_idx_file = root / TRAIN_INDICES_PATH

    if not dataset_file.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {dataset_file}")
    if not train_idx_file.exists():
        raise FileNotFoundError(f"Train split indices not found at {train_idx_file}")

    import numpy as np
    import pandas as pd
    from src.pipeline import create_pipeline_c, get_classifier

    train_indices = np.load(train_idx_file)
    if len(train_indices) != EXPECTED_TRAIN_RECORD_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_RECORD_COUNT} training indices, found {len(train_indices)}"
        )

    df = pd.read_csv(dataset_file)
    X_train = df.iloc[train_indices]["text"].to_numpy()
    y_train = df.iloc[train_indices]["label"].to_numpy().astype(int)

    classifier = get_classifier(
        "linear_svc",
        C=10.0,
        loss="squared_hinge",
        dual="auto",
        max_iter=5000,
    )
    pipeline_c = create_pipeline_c(classifier=classifier, heuristic_weight=0.05)
    pipeline_c.fit(X_train, y_train)

    return pipeline_c


def get_model_specs() -> Dict[str, Dict[str, Any]]:
    """Return immutable specifications and benchmark metadata for both available pipelines."""
    return {
        "pipeline_b": {
            "key": "pipeline_b",
            "name": "Pipeline B + LinearSVC",
            "display_name": "Pipeline B + LinearSVC (Production Champion)",
            "role": "Production Champion",
            "description": "Standard NLP text normalization (stopwords, Porter stemming) + sublinear TF-IDF (1-2 n-grams) + LinearSVC.",
            "hyperparameters": {
                "C": 2.0,
                "loss": "hinge",
                "dual": True,
                "max_iter": 5000,
            },
            "default_threshold": PIPELINE_B_DEFAULT_THRESHOLD,
            "calibrated_threshold": PIPELINE_B_CALIBRATED_THRESHOLD,
            "test_f1": 0.9684,
            "test_precision": 0.9787,
            "test_recall": 0.9583,
            "test_fpr": 0.0037,
            "test_ap": 0.9978,
            "latency_ms": 5.479,
            "uses_heuristics": False,
        },
        "pipeline_c": {
            "key": "pipeline_c",
            "name": "Pipeline C + LinearSVC",
            "display_name": "Pipeline C + LinearSVC (High-Throughput Alternative)",
            "role": "High-Throughput Alternative (2.72x Faster)",
            "description": "Domain-aware text normalization (entity placeholders) + TF-IDF combined with variance-scaled email heuristics (weight=0.05) + LinearSVC.",
            "hyperparameters": {
                "C": 10.0,
                "loss": "squared_hinge",
                "dual": "auto",
                "heuristic_weight": 0.05,
                "max_iter": 5000,
            },
            "default_threshold": PIPELINE_C_DEFAULT_THRESHOLD,
            "calibrated_threshold": PIPELINE_C_CALIBRATED_THRESHOLD,
            "test_f1": 0.9792,
            "test_precision": 0.9792,
            "test_recall": 0.9792,
            "test_fpr": 0.0037,
            "test_ap": 0.9974,
            "latency_ms": 2.016,
            "uses_heuristics": True,
        },
    }


def extract_heuristic_features(text: str) -> Dict[str, Any]:
    """Extract and format the 6 structural heuristic features used in Pipeline C.

    Features extracted:
    1. uppercase_ratio: Ratio of uppercase letters to total alphabetic characters.
    2. all_caps_word_count: Count of words with >=3 uppercase letters.
    3. exclamation_count: Total occurrences of '!'.
    4. question_count: Total occurrences of '?'.
    5. currency_symbol_count: Total occurrences of currency symbols ($ € £ ¥).
    6. url_count: Total count of explicit URLs.
    """
    from src.preprocessing import EmailHeuristicFeatures

    transformer = EmailHeuristicFeatures()
    raw_values = transformer.transform([text])[0]

    return {
        "uppercase_ratio": {
            "name": "Uppercase Ratio",
            "value": float(raw_values[0]),
            "formatted": f"{raw_values[0]:.2%}",
            "description": "Ratio of uppercase letters to alphabetic characters.",
        },
        "all_caps_word_count": {
            "name": "ALL-CAPS Words",
            "value": int(raw_values[1]),
            "formatted": str(int(raw_values[1])),
            "description": "Count of words with >= 3 consecutive capital letters.",
        },
        "exclamation_count": {
            "name": "Exclamation Marks",
            "value": int(raw_values[2]),
            "formatted": str(int(raw_values[2])),
            "description": "Total count of '!' exclamation marks.",
        },
        "question_count": {
            "name": "Question Marks",
            "value": int(raw_values[3]),
            "formatted": str(int(raw_values[3])),
            "description": "Total count of '?' question marks.",
        },
        "currency_symbol_count": {
            "name": "Currency Symbols",
            "value": int(raw_values[4]),
            "formatted": str(int(raw_values[4])),
            "description": "Count of currency symbols ($ € £ ¥).",
        },
        "url_count": {
            "name": "URL Count",
            "value": int(raw_values[5]),
            "formatted": str(int(raw_values[5])),
            "description": "Count of explicit URLs or web links.",
        },
    }


def predict_decision_margin(model: Pipeline, text: str) -> float:
    """Compute the continuous signed geometric decision margin for an email text.

    LinearSVC outputs a signed distance to the hyperplane:
    - Positive values indicate evidence toward Spam (class 1).
    - Negative values indicate evidence toward Ham (class 0).
    - This is NOT a probability or confidence percentage.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    scores = model.decision_function([text])
    return float(scores[0])


def classify_email(
    model: Pipeline,
    text: str,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """Classify an email text using the specified model and decision threshold.

    Decision rule:
        prediction = 1 (Spam) if decision_margin >= threshold else 0 (Ham)

    Returns a comprehensive diagnostic dictionary including signed margin,
    threshold comparison, predicted class, and distance to threshold.
    """
    margin = predict_decision_margin(model, text)
    is_spam = bool(margin >= threshold)
    label = "Spam" if is_spam else "Ham (Not Spam)"
    distance_to_threshold = abs(margin - threshold)
    signed_distance = margin - threshold

    return {
        "decision_margin": margin,
        "threshold": float(threshold),
        "is_spam": is_spam,
        "label": label,
        "predicted_class": 1 if is_spam else 0,
        "distance_to_threshold": distance_to_threshold,
        "signed_distance": signed_distance,
    }

