# Copyright 2024
"""JAX-specific logging and metrics integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_logging import log_metrics_wrapper
from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
SummaryWriter = None
with catch_optional_imports():
    from tensorboardX import SummaryWriter


def log_metrics(metrics: dict[str, float], step: int, log_dir: str = "logs") -> JSONDict:
    """Log metrics for a JAX training run using TensorBoard.

    Args:
    ----
        metrics: A dictionary of metric names and their float values.
        step: The current training step.
        log_dir: Directory to save the TensorBoard logs.

    Returns:
    -------
        A dictionary containing logging metadata.

    """
    return log_metrics_wrapper(backend_name="jax", metrics=metrics, step=step, log_dir=log_dir, summary_writer_cls=SummaryWriter, extra_fields={"action": "log_metrics"})
