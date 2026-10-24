"""Tests for PyTorch logging."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.pytorch.logging as logging_module
from tests.conftest import requires_torch

pytestmark = requires_torch


def test_pytorch_logging_real_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify PyTorch log_metrics writes metrics via SummaryWriter and JSONL fallback."""
    mock_writer_instance = MagicMock()
    mock_writer_cls = MagicMock(return_value=mock_writer_instance)
    monkeypatch.setattr(logging_module, "SummaryWriter", mock_writer_cls)

    log_dir = str(tmp_path / "tb_logs")
    res = logging_module.log_metrics({"loss": 0.25, "lr": 1e-4}, step=5, log_dir=log_dir)
    assert res["backend"] == "pytorch"
    assert res["action"] == "log_metrics"
    assert res["status"] == "success"
    assert res["step"] == 5
    mock_writer_instance.add_scalar.assert_any_call("loss", 0.25, 5)
    mock_writer_instance.add_scalar.assert_any_call("lr", 1e-4, 5)
    mock_writer_instance.close.assert_called_once()

    monkeypatch.setattr(logging_module, "SummaryWriter", None)
    fallback_dir = tmp_path / "fallback_logs"
    res_fb = logging_module.log_metrics({"loss": 0.5}, step=2, log_dir=str(fallback_dir))
    assert res_fb["status"] == "backend_unavailable"
    fb_file = Path(str(res_fb["fallback_file"]))
    assert fb_file.exists()
    payload = json.loads(fb_file.read_text(encoding="utf-8").strip())
    assert payload["backend"] == "pytorch"
    assert payload["metrics"] == {"loss": 0.5}
