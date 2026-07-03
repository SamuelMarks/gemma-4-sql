"""MLX-specific logging and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
SummaryWriter = None
with catch_optional_imports():
    from mlx.utils.tensorboard import SummaryWriter


def log_metrics(metrics: dict[str, float], step: int, log_dir: str = "logs") -> JSONDict:
    """Log metrics using MLX TensorBoard tools.

    Args:
    ----
        metrics: Dictionary of metric names to values.
        step: The current training step.
        log_dir: Directory to save the TensorBoard logs.

    Returns:
    -------
        A dictionary confirming the logged metrics.

    """
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=log_dir)
        for k, v in metrics.items():
            writer.add_scalar(k, v, step)
        writer.close()
        status = "success"
    else:
        status = "mocked_missing_tensorboard"
    return {"backend": "mlx", "step": step, "metrics": metrics, "status": status, "log_dir": log_dir}
