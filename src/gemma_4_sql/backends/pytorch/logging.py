# Copyright 2024
"""PyTorch-specific logging and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_logging import log_metrics_wrapper
from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
SummaryWriter = None
with catch_optional_imports():
    from torch.utils.tensorboard import SummaryWriter


def log_metrics(metrics: dict[str, float], step: int, log_dir: str = "logs") -> JSONDict:
    """Log metrics using PyTorch TensorBoard tools.

    Args:
    ----
        metrics: Dictionary of metric names to values.
        step: The current training step.
        log_dir: Directory to save the TensorBoard logs.

    Returns:
    -------
        A dictionary confirming the logged metrics.

    """
    return log_metrics_wrapper(
        backend_name="pytorch",
        metrics=metrics,
        step=step,
        log_dir=log_dir,
        summary_writer_cls=SummaryWriter,
    )
