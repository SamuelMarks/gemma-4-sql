"""MaxText-specific logging and metrics integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_logging import (
    get_tensorboard_summary_writer,
    log_metrics_wrapper,
)
from gemma_4_sql.exceptions import DependencyMissingError

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONValue, LogMetricsResultDict

SummaryWriter: Any | None = get_tensorboard_summary_writer()


def log_metrics(metrics: dict[str, float], step: int, log_dir: str = "logs", **kwargs: JSONValue) -> LogMetricsResultDict:
    """Log metrics for a MaxText training run using TensorBoard or JSONL fallback.

    Args:
        metrics: The evaluation or training metrics.
        step: The current training or logging step.
        log_dir: The directory to save logs.
        **kwargs: Additional logging keyword arguments (such as `allow_jsonl_fallback=True`).

    Returns:
        Structured `LogMetricsResultDict` containing logged metrics and status.

    Raises:
        DependencyMissingError: If TensorBoardX dependencies are missing and `allow_jsonl_fallback` is False.

    """
    if SummaryWriter is None and not kwargs.get("allow_jsonl_fallback"):
        raise DependencyMissingError("TensorBoardX dependencies are missing.", backend="maxtext")
    return log_metrics_wrapper(
        backend_name="maxtext",
        metrics=metrics,
        step=step,
        log_dir=log_dir,
        summary_writer_cls=SummaryWriter,
        extra_fields={"action": "log_metrics", "metrics": metrics, "logged_metrics": metrics},
    )
