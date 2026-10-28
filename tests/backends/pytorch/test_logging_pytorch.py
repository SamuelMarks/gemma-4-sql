"""Tests for PyTorch logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gemma_4_sql.backends.pytorch.logging as logging_module
from tests.conftest import requires_torch

pytestmark = requires_torch


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


def test_pytorch_logging_real_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify PyTorch log_metrics writes metrics via SummaryWriter and JSONL fallback."""
    writers: list[_RecordingSummaryWriter] = []

    def _factory(log_dir: str = "") -> _RecordingSummaryWriter:
        """Construct and record a `_RecordingSummaryWriter`."""
        w = _RecordingSummaryWriter(log_dir=log_dir)
        writers.append(w)
        return w

    monkeypatch.setattr(logging_module, "SummaryWriter", _factory)

    log_dir = str(tmp_path / "tb_logs")
    res = logging_module.log_metrics({"loss": 0.25, "lr": 1e-4}, step=5, log_dir=log_dir)
    assert res["backend"] == "pytorch"
    assert res["action"] == "log_metrics"
    assert res["status"] == "success"
    assert res["step"] == 5
    assert len(writers) == 1
    assert ("loss", 0.25, 5) in writers[0].scalars
    assert ("lr", 1e-4, 5) in writers[0].scalars
    assert writers[0].closed is True

    monkeypatch.setattr(logging_module, "SummaryWriter", None)
    fallback_dir = tmp_path / "fallback_logs"
    res_fb = logging_module.log_metrics({"loss": 0.5}, step=2, log_dir=str(fallback_dir))
    assert res_fb["status"] == "backend_unavailable"
    fb_file = Path(str(res_fb["fallback_file"]))
    assert fb_file.exists()
    payload = json.loads(fb_file.read_text(encoding="utf-8").strip())
    assert payload["backend"] == "pytorch"
    assert payload["metrics"] == {"loss": 0.5}
