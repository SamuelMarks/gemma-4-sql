"""MLX-specific logging and metrics."""

from __future__ import annotations

from gemma_4_sql.backends.common_logging import get_torch_tensorboard_writer, log_metrics_wrapper
from gemma_4_sql.type_hints import JSONValue, LogMetricsResultDict

SummaryWriter = get_torch_tensorboard_writer()


def log_metrics(metrics: dict[str, float], step: int, log_dir: str = "logs", **_kwargs: JSONValue) -> LogMetricsResultDict:
    """Log metrics using MLX TensorBoard tools.

    Args:
        metrics: The evaluation or training metrics.
        step: The current training or logging step.
        log_dir: The directory to save logs.
        **_kwargs: Additional logging keyword arguments.

    Returns:
        A dictionary containing the results.

    """
    return log_metrics_wrapper(
        backend_name="mlx",
        metrics=metrics,
        step=step,
        log_dir=log_dir,
        summary_writer_cls=SummaryWriter,
    )
