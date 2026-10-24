"""Unit and behavioral tests for logging jax."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.jax.logging as log_mod
from gemma_4_sql.backends.jax.logging import log_metrics
from tests.conftest import requires_jax

pytestmark = requires_jax


def test_log_metrics_behavioral(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify JAX log_metrics executes through log_metrics_wrapper with SummaryWriter and JSONL fallback."""
    mock_writer_instance = MagicMock()
    mock_writer_cls = MagicMock(return_value=mock_writer_instance)
    monkeypatch.setattr(log_mod, "SummaryWriter", mock_writer_cls)

    log_dir = str(tmp_path / "jax_logs")
    res = log_metrics({"loss": 0.5}, 10, log_dir)
    assert res["backend"] == "jax"
    assert res["action"] == "log_metrics"
    assert res["status"] == "success"
    assert res["step"] == 10
    mock_writer_instance.add_scalar.assert_called_once_with("loss", 0.5, 10)
    mock_writer_instance.close.assert_called_once()

    monkeypatch.setattr(log_mod, "SummaryWriter", None)
    res_fb = log_metrics({"loss": 0.5}, 11, str(tmp_path / "jax_fb"))
    assert res_fb["status"] == "backend_unavailable"
    fb_path = Path(str(res_fb["fallback_file"]))
    assert fb_path.exists()
    record = json.loads(fb_path.read_text(encoding="utf-8").strip())
    assert record["backend"] == "jax"
    assert record["step"] == 11
