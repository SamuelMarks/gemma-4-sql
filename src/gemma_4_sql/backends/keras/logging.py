"""Keras-specific logging and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_logging import log_metrics_wrapper
from gemma_4_sql.backends.lazy_loader import optional_import

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONValue, LogMetricsResultDict

tf = optional_import("tensorflow")


class _TFSummaryWriterAdapter:
    """Adapter wrapping `tf.summary.create_file_writer` to satisfy `SummaryWriter` protocol."""

    def __init__(self, log_dir: str = "logs") -> None:
        """Initialize the TensorFlow file writer for `log_dir`.

        Args:
            log_dir: Directory path where TensorFlow event files are written.

        """
        self._writer = tf.summary.create_file_writer(log_dir)

    def add_scalar(self, key: str, val: float, step: int) -> None:
        """Write a scalar summary metric at `step`.

        Args:
            key: Metric name.
            val: Metric scalar value.
            step: Training step index.

        """
        with self._writer.as_default():
            tf.summary.scalar(key, val, step=step)

    def close(self) -> None:
        """Close the underlying TensorFlow summary writer."""
        self._writer.close()


def log_metrics(metrics: dict[str, float], step: int, log_dir: str = "logs", **_kwargs: JSONValue) -> LogMetricsResultDict:
    """Log metrics using Keras/TensorFlow TensorBoard tools and common JSONL fallback.

    Args:
        metrics: The evaluation or training metrics.
        step: The current training or logging step.
        log_dir: The directory to save logs.
        **_kwargs: Additional logging keyword arguments.

    Returns:
        Structured `LogMetricsResultDict` containing logged metrics and status.

    Raises:
        DependencyMissingError: If TensorFlow dependencies are missing.

    """
    if tf is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("TensorFlow dependencies are missing.", backend="keras")

    has_summary = hasattr(tf, "summary")
    writer_cls: Any = _TFSummaryWriterAdapter if has_summary else None
    res = log_metrics_wrapper(
        backend_name="keras",
        metrics=metrics,
        step=step,
        log_dir=log_dir,
        summary_writer_cls=writer_cls,
        extra_fields={"metrics": metrics},
    )
    if not has_summary:
        res["status"] = "missing_summary_attr"
    return res
