"""
Keras-specific logging and metrics.
"""

from __future__ import annotations

from typing import Any

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def log_metrics(metrics: dict[str, float], step: int, log_dir: str = "logs") -> dict[str, Any]:
    """
    Logs metrics using Keras/TensorFlow TensorBoard tools.

    Args:
        metrics: Dictionary of metric names to values.
        step: The current training step.
        log_dir: Directory to save the TensorBoard logs.

    Returns:
        A dictionary confirming the logged metrics.
    """
    if tf is not None and hasattr(tf, "summary"):
        writer = tf.summary.create_file_writer(log_dir)
        with writer.as_default():
            for k, v in metrics.items():
                tf.summary.scalar(k, v, step=step)
        writer.close()
        status = "success"
    else:
        status = "mocked_missing_tensorboard"

    return {
        "backend": "keras",
        "step": step,
        "metrics": metrics,
        "status": status,
        "log_dir": log_dir,
    }
