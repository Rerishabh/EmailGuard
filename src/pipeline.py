"""Scikit-learn pipeline definitions combining feature extraction and classifiers.

This module provides factory functions for building reproducible, end-to-end
Scikit-Learn pipelines for email spam classification across three experimental
topologies:
    - Pipeline A: Minimal text normalization + TF-IDF vectorization.
    - Pipeline B: Standard NLP text normalization (stopwords, stemming) + TF-IDF.
    - Pipeline C: Domain-aware text normalization + TF-IDF combined with
                  variance-scaled email heuristic features via FeatureUnion.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import (
    DomainAwareTextTransformer,
    EmailHeuristicFeatures,
    MinimalTextTransformer,
    StandardNLPTextTransformer,
)


def _ensure_csr(X) -> csr_matrix:
    """Ensure sparse output is in CSR format (pickle-safe for joblib)."""
    return csr_matrix(X)


def _scale_heuristics(X, weight: float = 1.0):
    """Scale heuristic features by a positive scalar weight.

    Preserves sparse matrix format and does not mutate the input matrix.
    """
    weight_float = float(weight)
    if not np.isfinite(weight_float) or weight_float <= 0.0:
        raise ValueError(f"heuristic_weight must be a finite, strictly positive float, got {weight}")
    return X * weight_float


def build_text_vectorizer(
    ngram_range=(1, 2),
    min_df: int = 3,
    sublinear_tf: bool = True,
    **kwargs: Any,
) -> TfidfVectorizer:
    """Returns a TfidfVectorizer instantiated with verified Stage 5 baseline parameters.

    Accepts overrides via **kwargs.
    """
    params: Dict[str, Any] = {
        "ngram_range": ngram_range,
        "min_df": min_df,
        "sublinear_tf": sublinear_tf,
    }
    params.update(kwargs)
    return TfidfVectorizer(**params)


def get_classifier(classifier_name: str, **kwargs: Any) -> BaseEstimator:
    """Returns an un-fitted classifier instance with sensible default baselines.

    Supported classifier keys:
        - 'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
        - 'multinomial_nb': MultinomialNB(alpha=1.0)
        - 'decision_tree': DecisionTreeClassifier(max_depth=20, random_state=42)
        - 'random_forest': RandomForestClassifier(n_estimators=100, max_depth=25, random_state=42, n_jobs=-1)
        - 'linear_svc': LinearSVC(dual='auto', max_iter=2000, random_state=42)
        - 'knn': KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1)

    Merges any passed **kwargs to override default parameters.
    Raises ValueError for unrecognized classifier names.
    """
    key = classifier_name.lower().strip()

    defaults: Dict[str, Dict[str, Any]] = {
        "logistic_regression": {
            "max_iter": 1000,
            "random_state": 42,
        },
        "multinomial_nb": {
            "alpha": 1.0,
        },
        "decision_tree": {
            "max_depth": 20,
            "random_state": 42,
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 25,
            "random_state": 42,
            "n_jobs": -1,
        },
        "linear_svc": {
            "dual": "auto",
            "max_iter": 2000,
            "random_state": 42,
        },
        "knn": {
            "n_neighbors": 5,
            "metric": "cosine",
            "n_jobs": -1,
        },
    }

    if key not in defaults:
        valid_keys = sorted(list(defaults.keys()))
        raise ValueError(
            f"Unrecognized classifier '{classifier_name}'. Supported classifiers: {valid_keys}"
        )

    params = defaults[key].copy()
    params.update(kwargs)

    if key == "logistic_regression":
        return LogisticRegression(**params)
    elif key == "multinomial_nb":
        return MultinomialNB(**params)
    elif key == "decision_tree":
        return DecisionTreeClassifier(**params)
    elif key == "random_forest":
        return RandomForestClassifier(**params)
    elif key == "linear_svc":
        return LinearSVC(**params)
    elif key == "knn":
        return KNeighborsClassifier(**params)
    else:  # pragma: no cover
        raise ValueError(f"Unhandled classifier key: {key}")


def create_pipeline_a(
    classifier: Optional[BaseEstimator] = None,
    vectorizer_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Create Pipeline A (Minimal Normalization + TF-IDF Vectorizer).

    Text branch: MinimalTextTransformer() -> TfidfVectorizer(**vectorizer_params).
    If classifier is provided:
        Returns Pipeline([('transformer', MinimalTextTransformer()), ('tfidf', vec), ('classifier', classifier)]).
    If classifier is None:
        Returns Pipeline([('transformer', MinimalTextTransformer()), ('tfidf', vec)]).
    """
    v_params = vectorizer_params.copy() if vectorizer_params else {}
    vec = build_text_vectorizer(**v_params)
    steps: List[tuple] = [
        ("transformer", MinimalTextTransformer()),
        ("tfidf", vec),
    ]
    if classifier is not None:
        steps.append(("classifier", classifier))
    return Pipeline(steps)


def create_pipeline_b(
    classifier: Optional[BaseEstimator] = None,
    vectorizer_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Create Pipeline B (Standard NLP Normalization + TF-IDF Vectorizer).

    Text branch: StandardNLPTextTransformer() -> TfidfVectorizer(**vectorizer_params).
    If classifier is provided:
        Returns Pipeline([('transformer', StandardNLPTextTransformer()), ('tfidf', vec), ('classifier', classifier)]).
    If classifier is None:
        Returns Pipeline([('transformer', StandardNLPTextTransformer()), ('tfidf', vec)]).
    """
    v_params = vectorizer_params.copy() if vectorizer_params else {}
    vec = build_text_vectorizer(**v_params)
    steps: List[tuple] = [
        ("transformer", StandardNLPTextTransformer()),
        ("tfidf", vec),
    ]
    if classifier is not None:
        steps.append(("classifier", classifier))
    return Pipeline(steps)


def create_pipeline_c(
    classifier: Optional[BaseEstimator] = None,
    vectorizer_params: Optional[Dict[str, Any]] = None,
    heuristic_weight: float = 1.0,
) -> Pipeline:
    """Create Pipeline C (Domain-Aware TF-IDF + Scaled Heuristics via FeatureUnion).

    Branch 1: DomainAwareTextTransformer() -> TfidfVectorizer(**vectorizer_params).
    Branch 2: EmailHeuristicFeatures() -> StandardScaler(with_mean=False) -> FunctionTransformer(_scale_heuristics).
    FeatureUnion combining Branch 1 ('tfidf_branch') and Branch 2 ('heuristic_branch').
    Deterministic CSR conversion step using named top-level function:
        FunctionTransformer(_ensure_csr, accept_sparse=True, validate=False)
    If classifier is provided:
        Returns Pipeline([('features', union), ('to_csr', to_csr_step), ('classifier', classifier)]).
    If classifier is None:
        Returns Pipeline([('features', union), ('to_csr', to_csr_step)]).
    """
    v_params = vectorizer_params.copy() if vectorizer_params else {}
    vec = build_text_vectorizer(**v_params)

    weight_transformer = FunctionTransformer(
        _scale_heuristics,
        kw_args={"weight": float(heuristic_weight)},
        accept_sparse=True,
        validate=False,
    )

    tfidf_branch = Pipeline([
        ("transformer", DomainAwareTextTransformer()),
        ("tfidf", vec),
    ])

    heuristic_branch = Pipeline([
        ("features", EmailHeuristicFeatures()),
        ("scaler", StandardScaler(with_mean=False)),
        ("weight", weight_transformer),
    ])

    union = FeatureUnion([
        ("tfidf_branch", tfidf_branch),
        ("heuristic_branch", heuristic_branch),
    ])

    to_csr_step = FunctionTransformer(_ensure_csr, accept_sparse=True, validate=False)

    steps: List[tuple] = [
        ("features", union),
        ("to_csr", to_csr_step),
    ]
    if classifier is not None:
        steps.append(("classifier", classifier))
    return Pipeline(steps)


def extract_feature_names(
    pipeline_or_union: Union[Pipeline, FeatureUnion],
) -> np.ndarray:
    """Extract ordered output feature names from a fitted pipeline or FeatureUnion.

    Defensively verifies that underlying feature extractors are fitted before
    attempting name retrieval; raises ValueError if un-fitted.

    For Pipeline A & B:
        Retrieves named_steps['tfidf'].get_feature_names_out().
    For Pipeline C:
        Accesses branches via named_steps['features'].transformer_list (or directly if FeatureUnion).
        Extracts from 'tfidf_branch': named_steps['tfidf'].get_feature_names_out().
        Extracts from 'heuristic_branch': named_steps['features'].get_feature_names_out().
        Prefixes branch namespaces explicitly:
            [f"tfidf_branch__{f}" for f in tfidf_names]
            [f"heuristic_branch__{h}" for h in heur_names]
        Concatenates into a 1D NumPy array of string feature names.

    Raises:
        ValueError: If underlying feature extractors are not fitted, or if the
                    provided pipeline structure is unrecognized.
    """
    # Case 1: Pipeline A or Pipeline B (contains 'tfidf' step directly)
    if isinstance(pipeline_or_union, Pipeline) and "tfidf" in pipeline_or_union.named_steps:
        vec = pipeline_or_union.named_steps["tfidf"]
        if not hasattr(vec, "vocabulary_"):
            raise ValueError("Underlying feature extractors must be fitted before extracting feature names.")
        return np.asarray(vec.get_feature_names_out(), dtype=object)

    # Case 2: Pipeline C (Pipeline containing 'features' step) or direct FeatureUnion
    union_obj: Optional[FeatureUnion] = None
    if isinstance(pipeline_or_union, Pipeline) and "features" in pipeline_or_union.named_steps:
        features_step = pipeline_or_union.named_steps["features"]
        if isinstance(features_step, FeatureUnion):
            union_obj = features_step
    elif isinstance(pipeline_or_union, FeatureUnion):
        union_obj = pipeline_or_union

    if union_obj is not None:
        branches = dict(union_obj.transformer_list)
        if "tfidf_branch" in branches and "heuristic_branch" in branches:
            tfidf_pipe = branches["tfidf_branch"]
            heuristic_pipe = branches["heuristic_branch"]

            # Validate fitted status on tfidf step
            if not hasattr(tfidf_pipe, "named_steps") or "tfidf" not in tfidf_pipe.named_steps:
                raise ValueError("Underlying feature extractors must be fitted before extracting feature names.")
            vec = tfidf_pipe.named_steps["tfidf"]
            if not hasattr(vec, "vocabulary_"):
                raise ValueError("Underlying feature extractors must be fitted before extracting feature names.")

            # Validate fitted status on scaler step if present
            if hasattr(heuristic_pipe, "named_steps") and "scaler" in heuristic_pipe.named_steps:
                scaler = heuristic_pipe.named_steps["scaler"]
                if not hasattr(scaler, "scale_"):
                    raise ValueError("Underlying feature extractors must be fitted before extracting feature names.")

            # Extract names from tfidf branch
            tfidf_names = vec.get_feature_names_out()

            # Extract names from heuristic branch
            if not hasattr(heuristic_pipe, "named_steps") or "features" not in heuristic_pipe.named_steps:
                raise ValueError("Underlying feature extractors must be fitted before extracting feature names.")
            heur_step = heuristic_pipe.named_steps["features"]
            heur_names = heur_step.get_feature_names_out()

            # Explicitly validate heuristic names (stateless transformer: no is_fitted_ attribute needed)
            _EXPECTED_HEURISTIC_NAMES = [
                "uppercase_ratio",
                "all_caps_word_count",
                "exclamation_count",
                "question_count",
                "currency_symbol_count",
                "url_count",
            ]
            if list(heur_names) != _EXPECTED_HEURISTIC_NAMES:
                raise ValueError(
                    f"EmailHeuristicFeatures.get_feature_names_out() returned unexpected names: "
                    f"{list(heur_names)}. Expected: {_EXPECTED_HEURISTIC_NAMES}"
                )

            prefixed_tfidf = [f"tfidf_branch__{f}" for f in tfidf_names]
            prefixed_heur = [f"heuristic_branch__{h}" for h in heur_names]
            return np.array(prefixed_tfidf + prefixed_heur, dtype=object)

    # If unrecognized topology
    raise ValueError(
        f"Unsupported or unrecognized pipeline topology: {type(pipeline_or_union)}"
    )
