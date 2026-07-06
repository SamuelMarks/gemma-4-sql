"""Keras-specific logging and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
tf = None
with catch_optional_imports():
    import tensorflow as tf


def log_metrics(metrics: dict[str, float], step: int, log_dir: str = "logs") -> JSONDict:
    """Log metrics using Keras/TensorFlow TensorBoard tools.

    Args:
        metrics: The evaluation or training metrics.
        step: The current training or logging step.
        log_dir: The directory to save logs.

    Returns:
        A dictionary containing the results.
    """
    if tf is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("TensorFlow dependencies are missing.")
    if hasattr(tf, "summary"):
        writer = tf.summary.create_file_writer(log_dir)
        with writer.as_default():
            for k, v in metrics.items():
                tf.summary.scalar(k, v, step=step)
        writer.close()
        status = "success"
    else:
        status = "missing_summary_attr"  # pragma: no cover
    return {"backend": "keras", "step": step, "metrics": metrics, "status": status, "log_dir": log_dir}
