"""Common logging utility for backends."""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.lazy_loader import optional_attr
from gemma_4_sql.type_hints import JSONValue, LogMetricsResultDict


@runtime_checkable
class SummaryWriterInstanceProtocol(Protocol):
    """Protocol for a TensorBoard `SummaryWriter` instance."""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """Record a scalar metric at `global_step`."""
        ...


@runtime_checkable
class ClosableSummaryWriterProtocol(SummaryWriterInstanceProtocol, Protocol):
    """Protocol for a TensorBoard `SummaryWriter` instance supporting `.close()`."""

    def close(self) -> None:
        """Flush and close the summary writer."""
        ...


type SummaryWriterFactory = Callable[..., SummaryWriterInstanceProtocol] | type


def get_tensorboard_summary_writer() -> type | None:
    """Resolve `tensorboardX.SummaryWriter` lazily if installed.

    Returns:
        The `SummaryWriter` class or `None` if `tensorboardX` is unavailable.

    """
    writer_cls = optional_attr("tensorboardX", "SummaryWriter", type)
    return writer_cls if isinstance(writer_cls, type) else None


def get_torch_tensorboard_writer() -> type | None:
    """Resolve `torch.utils.tensorboard.SummaryWriter` lazily if installed.

    Returns:
        The `SummaryWriter` class or `None` if `torch.utils.tensorboard` is unavailable.

    """
    writer_cls = optional_attr("torch.utils.tensorboard", "SummaryWriter", type)
    return writer_cls if isinstance(writer_cls, type) else None


def log_metrics_wrapper(
    backend_name: str,
    metrics: dict[str, float],
    step: int,
    log_dir: str,
    summary_writer_cls: SummaryWriterFactory | None,
    extra_fields: Mapping[str, JSONValue] | None = None,
    step_duration_s: float | None = None,
) -> LogMetricsResultDict:
    """Log metrics for a training run using TensorBoard or a structured file-based fallback.

    When TensorBoard is unavailable, logs are automatically appended as JSON-lines
    to `{log_dir}/metrics.jsonl` to ensure persistence across all backend environments.

    Args:
        backend_name: The backend framework to use.
        metrics: The evaluation or training metrics mapping name to float values.
        step: The current training or logging step.
        log_dir: The directory to save logs and metric records.
        summary_writer_cls: Optional TensorBoard SummaryWriter class.
        extra_fields: Optional mapping representing extra string fields.
        step_duration_s: Optional duration of the logged step in seconds.

    Returns:
        A dictionary containing the results and status.

    """
    timestamp = time.time()
    fallback_file: str | None = None

    if summary_writer_cls is not None:
        writer: SummaryWriterInstanceProtocol = summary_writer_cls(log_dir=log_dir)
        try:
            for k, v in metrics.items():
                writer.add_scalar(k, v, step)
        finally:
            with contextlib.suppress(AttributeError):
                writer.close()  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        status = "success"
    else:
        status = "backend_unavailable"
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        metrics_file = log_path / "metrics.jsonl"
        record: dict[str, JSONValue] = {
            "backend": backend_name,
            "step": step,
            "timestamp": timestamp,
            "metrics": metrics,
        }
        if step_duration_s is not None:
            record["step_duration_s"] = step_duration_s
        if extra_fields:
            record.update(extra_fields)
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + chr(10))
        fallback_file = str(metrics_file)

    result: LogMetricsResultDict = {
        "backend": backend_name,
        "action": "log_metrics",
        "step": step,
        "logged_step": step,
        "metrics": metrics,
        "logged_metrics": metrics,
        "status": status,
        "log_dir": log_dir,
        "timestamp": timestamp,
    }

    if step_duration_s is not None and step_duration_s > 0:
        result["step_duration_s"] = step_duration_s
    if fallback_file is not None:
        result["fallback_file"] = fallback_file
    if extra_fields:
        if "run_id" in extra_fields and isinstance(extra_fields["run_id"], str):
            result["run_id"] = extra_fields["run_id"]
        if "model" in extra_fields and isinstance(extra_fields["model"], str):
            result["model"] = extra_fields["model"]

    return result


__all__ = [
    "ClosableSummaryWriterProtocol",
    "SummaryWriterFactory",
    "SummaryWriterInstanceProtocol",
    "get_tensorboard_summary_writer",
    "get_torch_tensorboard_writer",
    "log_metrics_wrapper",
]
