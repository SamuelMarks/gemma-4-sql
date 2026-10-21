"""SDK Logging module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.sdk.registry import get_backend

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONValue, LogMetricsResultDict


def log_metrics(
    metrics: dict[str, float],
    step: int,
    log_dir: str = "logs",
    backend: str = "jax",
    **kwargs: JSONValue,
) -> LogMetricsResultDict:
    """Log training or evaluation metrics to TensorBoard.

    Args:
        metrics: The evaluation or training metrics.
        step: The current training or logging step.
        log_dir: The directory to save logs.
        backend: The backend framework to use.
        **kwargs: Additional backend logging options (e.g., `step_duration_s`).

    Returns:
        Structured `LogMetricsResultDict` containing `backend`, `step`, `logged_metrics`, `log_dir`, `timestamp`, and `status`.

    """
    return get_backend(backend).log_metrics(metrics, step, log_dir, **kwargs)


__all__ = ["log_metrics"]
