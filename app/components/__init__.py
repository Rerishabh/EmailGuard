"""UI Components package for Email Spam Detection Streamlit Application."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "get_email_examples": ".email_examples",
    "render_prediction_card": ".prediction",
    "generate_prediction_card_html": ".prediction",
    "render_margin_gauge": ".visualization",
    "generate_margin_gauge_svg": ".visualization",
    "render_dataset_summary": ".overview",
    "render_project_overview": ".overview",
    "render_benchmark_tab": ".benchmark",
    "render_evaluation_tab": ".evaluation",
    "render_methodology_tab": ".methodology",
}

__all__ = [
    "get_email_examples",
    "render_prediction_card",
    "render_margin_gauge",
    "generate_margin_gauge_svg",
    "render_dataset_summary",
    "render_project_overview",
    "render_benchmark_tab",
    "render_evaluation_tab",
    "render_methodology_tab",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve exported component symbols to avoid eager heavy imports at startup."""
    module_rel = _EXPORTS.get(name)
    if module_rel is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = importlib.import_module(module_rel, package=__package__)
    except (ImportError, ValueError):
        module_name = module_rel.lstrip(".")
        module = importlib.import_module(f"app.components.{module_name}")

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
