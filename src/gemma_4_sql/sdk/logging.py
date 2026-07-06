"""SDK Logging module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict


def log_metrics(metrics: dict[str, float], step: int, log_dir: str = "logs", backend: str = "jax") -> JSONDict:
    """Log training or evaluation metrics to TensorBoard.

    Args:
        metrics: The evaluation or training metrics.
        step: The current training or logging step.
        log_dir: The directory to save logs.
        backend: The backend framework to use.

    Returns:
        A dictionary containing the results.
    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).log_metrics(metrics, step, log_dir)
