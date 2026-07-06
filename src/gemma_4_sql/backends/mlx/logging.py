"""MLX-specific logging and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_logging import log_metrics_wrapper
from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
SummaryWriter = None
with catch_optional_imports():
    from mlx.utils.tensorboard import SummaryWriter


def log_metrics(metrics: dict[str, float], step: int, log_dir: str = "logs") -> JSONDict:
    """Log metrics using MLX TensorBoard tools.

    Args:
        metrics: The evaluation or training metrics.
        step: The current training or logging step.
        log_dir: The directory to save logs.

    Returns:
        A dictionary containing the results.
    """
    return log_metrics_wrapper(  # pragma: no cover
        backend_name="mlx",
        metrics=metrics,
        step=step,
        log_dir=log_dir,
        summary_writer_cls=SummaryWriter,
    )
