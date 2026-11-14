"""Keras-specific logging and metrics."""

from __future__ import annotations

import types
from typing import Protocol, Self, runtime_checkable

from gemma_4_sql.backends.common_logging import SummaryWriterFactory, log_metrics_wrapper
from gemma_4_sql.backends.lazy_loader import ModuleNamespaceProtocol, optional_import
from gemma_4_sql.type_hints import JSONValue, LogMetricsResultDict


@runtime_checkable
class _WriterDefaultContextProtocol(Protocol):
    """Protocol for the context manager returned by `SummaryWriter.as_default()`."""

    def __enter__(self) -> Self | None:
        """Enter the summary writer context.

        Returns:
            Entered context manager or `None`.

        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
        /,
    ) -> bool | None:
        """Exit the summary writer context.

        Args:
            exc_type: Exception type if raised, else `None`.
            exc_val: Exception instance if raised, else `None`.
            exc_tb: Traceback if raised, else `None`.

        Returns:
            Suppression flag or `None`.

        """
        ...


@runtime_checkable
class _TFFileWriterProtocol(Protocol):
    """Protocol for a TensorFlow `SummaryWriter` returned by `tf.summary.create_file_writer`."""

    def as_default(self) -> _WriterDefaultContextProtocol:
        """Return a context manager activating this summary writer as default.

        Returns:
            Active summary writer context manager.

        """
        ...

    def close(self) -> None:
        """Flush and close the summary writer."""
        ...


@runtime_checkable
class _TFSummaryNamespaceProtocol(Protocol):
    """Protocol for `tf.summary` exposing `create_file_writer` and `scalar`."""

    def create_file_writer(self, logdir: str, /) -> _TFFileWriterProtocol:
        """Create a summary file writer for `logdir`.

        Args:
            logdir: Output directory path.

        Returns:
            Initialized `_TFFileWriterProtocol`.

        """
        ...

    def scalar(self, name: str, data: float, *, step: int) -> bool | None:
        """Record a scalar summary metric value.

        Args:
            name: Metric tag name.
            data: Scalar float value.
            step: Global step index.

        Returns:
            Optional boolean write indicator or `None`.

        """
        ...


@runtime_checkable
class _TFSummaryHolderProtocol(Protocol):
    """Protocol for a `tensorflow` module exposing `.summary`."""

    @property
    def summary(self) -> _TFSummaryNamespaceProtocol:
        """Provide `tf.summary` namespace."""
        ...


tf: _TFSummaryHolderProtocol | ModuleNamespaceProtocol | None = optional_import("tensorflow")


class _TFSummaryWriterAdapter:
    """Adapter wrapping `tf.summary.create_file_writer` to satisfy `SummaryWriter` protocol."""

    _writer: _TFFileWriterProtocol

    def __init__(self, log_dir: str = "logs") -> None:
        """Initialize the TensorFlow file writer for `log_dir`.

        Args:
            log_dir: Directory path where TensorFlow event files are written.

        """
        assert isinstance(tf, _TFSummaryHolderProtocol)
        self._writer = tf.summary.create_file_writer(log_dir)

    def add_scalar(self, key: str, val: float, step: int) -> None:
        """Write a scalar summary metric at `step`.

        Args:
            key: Metric name.
            val: Metric scalar value.
            step: Training step index.

        """
        assert isinstance(tf, _TFSummaryHolderProtocol)
        with self._writer.as_default():
            tf.summary.scalar(key, val, step=step)

    def close(self) -> None:
        """Close the underlying TensorFlow summary writer."""
        self._writer.close()


def log_metrics(
    metrics: dict[str, float],
    step: int,
    log_dir: str = "logs",
    **_kwargs: JSONValue,
) -> LogMetricsResultDict:
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

    has_summary = isinstance(tf, _TFSummaryHolderProtocol)
    writer_cls: SummaryWriterFactory | None = _TFSummaryWriterAdapter if has_summary else None
    res = log_metrics_wrapper(
        backend_name="keras",
        metrics=metrics,
        step=step,
        log_dir=log_dir,
        summary_writer_cls=writer_cls,
    )
    if not has_summary:
        res["status"] = "missing_summary_attr"
    return res
