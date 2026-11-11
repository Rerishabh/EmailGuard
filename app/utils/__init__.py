"""Utility modules for the Email Spam Detection Streamlit Application."""

try:
    from .model_loader import (
        CHAMPION_CONFIG_PATH,
        CHAMPION_MODEL_PATH,
        CLEANED_DATASET_PATH,
        PIPELINE_B_CALIBRATED_THRESHOLD,
        PIPELINE_B_DEFAULT_THRESHOLD,
        PIPELINE_C_CALIBRATED_THRESHOLD,
        PIPELINE_C_DEFAULT_THRESHOLD,
        TRAIN_INDICES_PATH,
        classify_email,
        extract_heuristic_features,
        get_model_specs,
        get_project_root,
        load_champion_config,
        load_champion_model,
        load_pipeline_c_model,
        predict_decision_margin,
    )
except (ImportError, ValueError):
    from app.utils.model_loader import (
        CHAMPION_CONFIG_PATH,
        CHAMPION_MODEL_PATH,
        CLEANED_DATASET_PATH,
        PIPELINE_B_CALIBRATED_THRESHOLD,
        PIPELINE_B_DEFAULT_THRESHOLD,
        PIPELINE_C_CALIBRATED_THRESHOLD,
        PIPELINE_C_DEFAULT_THRESHOLD,
        TRAIN_INDICES_PATH,
        classify_email,
        extract_heuristic_features,
        get_model_specs,
        get_project_root,
        load_champion_config,
        load_champion_model,
        load_pipeline_c_model,
        predict_decision_margin,
    )

__all__ = [
    "CHAMPION_CONFIG_PATH",
    "CHAMPION_MODEL_PATH",
    "CLEANED_DATASET_PATH",
    "PIPELINE_B_CALIBRATED_THRESHOLD",
    "PIPELINE_B_DEFAULT_THRESHOLD",
    "PIPELINE_C_CALIBRATED_THRESHOLD",
    "PIPELINE_C_DEFAULT_THRESHOLD",
    "TRAIN_INDICES_PATH",
    "classify_email",
    "extract_heuristic_features",
    "get_model_specs",
    "get_project_root",
    "load_champion_config",
    "load_champion_model",
    "load_pipeline_c_model",
    "predict_decision_margin",
]

