"""Tests for MLX logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gemma_4_sql.backends.mlx.logging as logging_module
from tests.conftest import requires_mlx

pytestmark = requires_mlx


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


def test_mlx_logging_real_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify MLX log_metrics writes metrics via SummaryWriter and JSONL fallback without sys.modules mutation."""
    writers: list[_RecordingSummaryWriter] = []

    def _factory(log_dir: str = "") -> _RecordingSummaryWriter:
        """Construct and record a `_RecordingSummaryWriter`."""
        w = _RecordingSummaryWriter(log_dir=log_dir)
        writers.append(w)
        return w

    monkeypatch.setattr(logging_module, "SummaryWriter", _factory)

    log_dir = str(tmp_path / "mlx_tb")
    res = logging_module.log_metrics({"loss": 0.75}, step=3, log_dir=log_dir)
    assert res["backend"] == "mlx"
    assert res["action"] == "log_metrics"
    assert res["status"] == "success"
    assert len(writers) == 1
    assert writers[0].scalars == [("loss", 0.75, 3)]
    assert writers[0].closed is True

    monkeypatch.setattr(logging_module, "SummaryWriter", None)
    fb_dir = tmp_path / "mlx_fb"
    res_fb = logging_module.log_metrics({"loss": 0.75}, step=4, log_dir=str(fb_dir))
    assert res_fb["status"] == "backend_unavailable"
    fb_file = Path(str(res_fb["fallback_file"]))
    assert fb_file.exists()
    data = json.loads(fb_file.read_text(encoding="utf-8").strip())
    assert data["backend"] == "mlx"
    assert data["step"] == 4
