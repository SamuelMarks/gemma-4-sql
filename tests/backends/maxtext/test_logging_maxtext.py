"""Unit and behavioral tests for logging maxtext."""

from pathlib import Path
from unittest.mock import patch

import pytest

from gemma_4_sql.backends.maxtext.logging import log_metrics
from gemma_4_sql.exceptions import DependencyMissingError


class _RecordingSummaryWriter:
    """Typed test double for `SummaryWriter` implementing `ClosableSummaryWriterProtocol`."""

    def __init__(self, log_dir: str = "") -> None:
        """Initialize the recording summary writer."""
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []
        self.closed = False

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """Record a scalar metric at `global_step`."""
        self.scalars.append((tag, scalar_value, global_step))

    def close(self) -> None:
        """Mark the summary writer as closed."""
        self.closed = True


def test_log_metrics_success() -> None:
    """Verify log metrics success behavior."""
    metrics = {"loss": 0.5, "accuracy": 0.9}
    step = 10
    log_dir = "test_logs"

    writers: list[_RecordingSummaryWriter] = []

    def _factory(log_dir: str = "") -> _RecordingSummaryWriter:
        """Construct and record a `_RecordingSummaryWriter`."""
        w = _RecordingSummaryWriter(log_dir=log_dir)
        writers.append(w)
        return w

    with patch("gemma_4_sql.backends.maxtext.logging.SummaryWriter", new=_factory):
        result = log_metrics(metrics, step, log_dir)

    assert result["backend"] == "maxtext"
    assert result.get("action") == "log_metrics"
    assert result.get("step") == step
    assert result.get("metrics") == metrics
    assert result.get("logged_metrics") == metrics
    assert result["status"] == "success"
    assert result["log_dir"] == log_dir
    assert "timestamp" in result

    assert len(writers) == 1
    assert writers[0].log_dir == log_dir
    assert ("loss", 0.5, step) in writers[0].scalars
    assert ("accuracy", 0.9, step) in writers[0].scalars
    assert writers[0].closed is True


def test_log_metrics_missing_dependency(tmp_path: Path) -> None:
    """Verify log metrics missing dependency behavior."""
    with patch("gemma_4_sql.backends.maxtext.logging.SummaryWriter", new=None):
        with pytest.raises(DependencyMissingError, match=r"TensorBoardX dependencies are missing\."):
            log_metrics({"loss": 0.5}, 10)
        fallback_res = log_metrics({"loss": 0.5}, 10, log_dir=str(tmp_path), allow_jsonl_fallback=True)
        assert fallback_res["status"] == "backend_unavailable"
        assert "fallback_file" in fallback_res
