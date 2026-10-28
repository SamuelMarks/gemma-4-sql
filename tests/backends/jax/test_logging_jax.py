"""Unit and behavioral tests for logging jax."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gemma_4_sql.backends.jax.logging as log_mod
from gemma_4_sql.backends.jax.logging import log_metrics
from tests.conftest import requires_jax

pytestmark = requires_jax


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


def test_log_metrics_behavioral(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify JAX log_metrics executes through log_metrics_wrapper with SummaryWriter and JSONL fallback."""
    created_writers: list[_RecordingSummaryWriter] = []

    def _writer_factory(log_dir: str = "") -> _RecordingSummaryWriter:
        """Create and record a `_RecordingSummaryWriter` instance."""
        writer = _RecordingSummaryWriter(log_dir=log_dir)
        created_writers.append(writer)
        return writer

    monkeypatch.setattr(log_mod, "SummaryWriter", _writer_factory)

    log_dir = str(tmp_path / "jax_logs")
    res = log_metrics({"loss": 0.5}, 10, log_dir)
    assert res["backend"] == "jax"
    assert res["action"] == "log_metrics"
    assert res["status"] == "success"
    assert res["step"] == 10
    assert len(created_writers) == 1
    assert created_writers[0].scalars == [("loss", 0.5, 10)]
    assert created_writers[0].closed is True

    monkeypatch.setattr(log_mod, "SummaryWriter", None)
    res_fb = log_metrics({"loss": 0.5}, 11, str(tmp_path / "jax_fb"))
    assert res_fb["status"] == "backend_unavailable"
    fb_path = Path(str(res_fb["fallback_file"]))
    assert fb_path.exists()
    record = json.loads(fb_path.read_text(encoding="utf-8").strip())
    assert record["backend"] == "jax"
    assert record["step"] == 11
