"""
JAX-specific logging and metrics integration.
"""

from __future__ import annotations

from typing import Any

try:
    from tensorboardX import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


def log_metrics(
    metrics: dict[str, float], step: int, log_dir: str = "logs"
) -> dict[str, Any]:
    """
    Logs metrics for a JAX training run using TensorBoard.

    Args:
        metrics: A dictionary of metric names and their float values.
        step: The current training step.
        log_dir: Directory to save the TensorBoard logs.

    Returns:
        A dictionary containing logging metadata.
    """
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=log_dir)
        for k, v in metrics.items():
            writer.add_scalar(k, v, step)
        writer.close()
        status = "success"
    else:
        status = "mocked_missing_tensorboard"

    return {
        "backend": "jax",
        "action": "log_metrics",
        "step": step,
        "metrics": metrics,
        "status": status,
        "log_dir": log_dir,
    }
