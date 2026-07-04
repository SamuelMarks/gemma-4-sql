# Copyright 2024
"""SDK Logging module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict


def log_metrics(metrics: dict[str, float], step: int, log_dir: str = "logs", backend: str = "jax") -> JSONDict:
    """Log training or evaluation metrics to TensorBoard.

    Args:
    ----
        metrics: A dictionary of metric names and their float values.
        step: The current training or evaluation step.
        log_dir: The directory to save the TensorBoard logs.
        backend: The backend framework ('jax', 'keras', 'maxtext', 'pytorch').

    Returns:
    -------
        Logging results dictionary.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).log_metrics(metrics, step, log_dir)
