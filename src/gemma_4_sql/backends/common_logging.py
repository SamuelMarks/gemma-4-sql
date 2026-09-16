"""Common logging utility for backends."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict


def log_metrics_wrapper(
    backend_name: str,
    metrics: dict[str, float],
    step: int,
    log_dir: str,
    summary_writer_cls: type | None,
    extra_fields: dict[str, str] | None = None,
    step_duration_s: float | None = None,
) -> JSONDict:
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
        writer = summary_writer_cls(log_dir=log_dir)
        try:
            for k, v in metrics.items():
                writer.add_scalar(k, v, step)
        finally:
            if hasattr(writer, "close"):
                writer.close()
        status = "success"
    else:
        status = "mocked_missing_tensorboard"
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        metrics_file = log_path / "metrics.jsonl"
        record: dict[str, Any] = {
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
            f.write(f"{json.dumps(record)}\n")
        fallback_file = str(metrics_file)

    result: JSONDict = {
        "backend": backend_name,
        "step": step,
        "metrics": metrics,
        "status": status,
        "log_dir": log_dir,
        "timestamp": timestamp,
    }

    if step_duration_s is not None:
        result["step_duration_s"] = step_duration_s
    if fallback_file is not None:
        result["fallback_file"] = fallback_file
    if extra_fields:
        result.update(extra_fields)

    return result
