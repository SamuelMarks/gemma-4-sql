"""Common logging utility for backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict


def log_metrics_wrapper(backend_name: str, metrics: dict[str, float], step: int, log_dir: str, summary_writer_cls: type | None, extra_fields: dict[str, str] | None = None) -> JSONDict:
    """Log metrics for a training run using a unified TensorBoard wrapper.

    Args:
        backend_name: The backend framework to use.
        metrics: The evaluation or training metrics.
        step: The current training or logging step.
        log_dir: The directory to save logs.
        summary_writer_cls: The summary writer cls.
        extra_fields: A mapping representing extra fields.

    Returns:
        A dictionary containing the results.
    """
    if summary_writer_cls is not None:
        writer = summary_writer_cls(log_dir=log_dir)
        for k, v in metrics.items():
            writer.add_scalar(k, v, step)
        if hasattr(writer, "close"):  # pragma: no cover
            writer.close()
        status = "success"
    else:
        status = "mocked_missing_tensorboard"

    result: JSONDict = {"backend": backend_name, "step": step, "metrics": metrics, "status": status, "log_dir": log_dir}

    if extra_fields:
        result.update(extra_fields)

    return result
