"""JAX-specific logging and metrics integration with multi-host rank-0 gating."""

from __future__ import annotations

import time

from gemma_4_sql.backends._jax_common import initialize_jax_distributed_if_configured
from gemma_4_sql.backends.common_logging import get_tensorboard_summary_writer, log_metrics_wrapper
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.exceptions import ExecutionStatus
from gemma_4_sql.type_hints import JSONValue, LogMetricsResultDict

SummaryWriter = get_tensorboard_summary_writer()


def log_metrics(
    metrics: dict[str, float],
    step: int,
    log_dir: str = "logs",
    **kwargs: JSONValue,
) -> LogMetricsResultDict:
    """Log metrics for a JAX training run on rank 0 using TensorBoard or JSONL fallback.

    Args:
        metrics: The evaluation or training metrics.
        step: The current training or logging step.
        log_dir: The directory to save logs.
        **kwargs: Additional logging keyword arguments (`backend_alias`).

    Returns:
        A dictionary containing the results.

    """
    backend_alias = str(kwargs.get("backend_alias", "jax"))
    jax_mod = optional_import("jax")
    topo = initialize_jax_distributed_if_configured(jax_mod)
    if topo.process_count > 1 and topo.process_index != 0:
        return {
            "backend": backend_alias,
            "action": "log_metrics",
            "step": step,
            "metrics": metrics,
            "status": "skipped_non_zero_rank",
            "status_code": ExecutionStatus.SUCCESS.value,
            "log_dir": log_dir,
            "timestamp": time.time(),
        }
    return log_metrics_wrapper(
        backend_name=backend_alias,
        metrics=metrics,
        step=step,
        log_dir=log_dir,
        summary_writer_cls=SummaryWriter,
    )
