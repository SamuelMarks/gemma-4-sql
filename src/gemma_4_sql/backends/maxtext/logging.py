"""
MaxText-specific logging and metrics integration.
"""

from __future__ import annotations

from typing import Any

def log_metrics(metrics: dict[str, float], step: int) -> dict[str, Any]:
    """
    Logs metrics for a MaxText training run.

    Args:
        metrics: A dictionary of metric names and their float values.
        step: The current training step.

    Returns:
        A dictionary containing logging metadata.
    """
    return {
        "backend": "maxtext",
        "action": "log_metrics",
        "step": step,
        "metrics": metrics,
        "status": "success",
    }
