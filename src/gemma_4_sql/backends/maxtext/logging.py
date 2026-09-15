"""MaxText-specific logging and metrics integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict

try:
    from tensorboardX import SummaryWriter as _SummaryWriter

    SummaryWriter: Any = _SummaryWriter
except (ImportError, AttributeError):
    SummaryWriter = None


def log_metrics(metrics: dict[str, float], step: int, log_dir: str = "logs") -> JSONDict:
    """Log metrics for a MaxText training run using TensorBoard.

    Args:
        metrics: The evaluation or training metrics.
        step: The current training or logging step.
        log_dir: The directory to save logs.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If TensorBoardX dependencies are missing.
    """
    if SummaryWriter is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("TensorBoardX dependencies are missing.")
    writer = SummaryWriter(log_dir=log_dir)
    for k, v in metrics.items():
        writer.add_scalar(k, v, step)
    writer.close()
    status = "success"
    return {"backend": "maxtext", "action": "log_metrics", "step": step, "metrics": metrics, "status": status, "log_dir": log_dir}
