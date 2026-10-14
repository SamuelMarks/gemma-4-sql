"""Keras-specific logging and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import optional_import

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONValue, LogMetricsResultDict

tf = optional_import("tensorflow")


def log_metrics(metrics: dict[str, float], step: int, log_dir: str = "logs", **_kwargs: JSONValue) -> LogMetricsResultDict:
    """Log metrics using Keras/TensorFlow TensorBoard tools.

    Args:
        metrics: The evaluation or training metrics.
        step: The current training or logging step.
        log_dir: The directory to save logs.
        **_kwargs: Additional logging keyword arguments.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If TensorFlow dependencies are missing.

    """
    if tf is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("TensorFlow dependencies are missing.")
    if hasattr(tf, "summary"):
        writer = tf.summary.create_file_writer(log_dir)
        try:
            with writer.as_default():
                for k, v in metrics.items():
                    tf.summary.scalar(k, v, step=step)
        finally:
            writer.close()
        status = "success"
    else:
        status = "missing_summary_attr"
    return {"backend": "keras", "step": step, "metrics": metrics, "status": status, "log_dir": log_dir}
