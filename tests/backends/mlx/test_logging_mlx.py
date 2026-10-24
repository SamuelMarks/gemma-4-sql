"""Tests for MLX logging."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.mlx.logging as logging_module
from tests.conftest import requires_mlx

pytestmark = requires_mlx


def test_mlx_logging_real_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify MLX log_metrics writes metrics via SummaryWriter and JSONL fallback without sys.modules mutation."""
    mock_writer_instance = MagicMock()
    mock_writer_cls = MagicMock(return_value=mock_writer_instance)
    monkeypatch.setattr(logging_module, "SummaryWriter", mock_writer_cls)

    log_dir = str(tmp_path / "mlx_tb")
    res = logging_module.log_metrics({"loss": 0.75}, step=3, log_dir=log_dir)
    assert res["backend"] == "mlx"
    assert res["action"] == "log_metrics"
    assert res["status"] == "success"
    mock_writer_instance.add_scalar.assert_called_once_with("loss", 0.75, 3)
    mock_writer_instance.close.assert_called_once()

    monkeypatch.setattr(logging_module, "SummaryWriter", None)
    fb_dir = tmp_path / "mlx_fb"
    res_fb = logging_module.log_metrics({"loss": 0.75}, step=4, log_dir=str(fb_dir))
    assert res_fb["status"] == "backend_unavailable"
    fb_file = Path(str(res_fb["fallback_file"]))
    assert fb_file.exists()
    data = json.loads(fb_file.read_text(encoding="utf-8").strip())
    assert data["backend"] == "mlx"
    assert data["step"] == 4
