"""
SDK Logging module.
"""

from __future__ import annotations

from typing import Any

def log_metrics(
    metrics: dict[str, float], step: int, backend: str = "jax"
) -> dict[str, Any]:
    """
    Logs training or evaluation metrics.

    Args:
        metrics: A dictionary of metric names and their float values.
        step: The current training or evaluation step.
        backend: The backend framework ('jax', 'keras', or 'maxtext').

    Returns:
        Logging results dictionary.
    """
    if backend == "jax":
        from gemma_4_sql.backends.jax.logging import log_metrics as jax_log

        return jax_log(metrics, step)
    elif backend == "keras":
        from gemma_4_sql.backends.keras.logging import log_metrics as keras_log

        return keras_log(metrics, step)
    elif backend == "maxtext":
        from gemma_4_sql.backends.maxtext.logging import log_metrics as maxtext_log

        return maxtext_log(metrics, step)
    elif backend == "pytorch":
        from gemma_4_sql.backends.pytorch.logging import (
            log_metrics as pytorch_log,
        )

        return pytorch_log(metrics, step)
    else:
        raise ValueError(f"Unknown backend: {backend}")
