"""MLX-specific logging and metrics."""

from __future__ import annotations

try:
    from mlx.utils.tensorboard import SummaryWriter
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    SummaryWriter = None


def log_metrics(metrics: dict[str, float], step: int, log_dir: str = "logs") -> dict[str, object]:
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
