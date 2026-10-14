"""Unit tests for the MLX backend package exports and logging wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gemma_4_sql.backends.mlx import __all__, get_trainer
from tests.conftest import requires_mlx

pytestmark = requires_mlx


def test_get_trainer() -> None:
    """Verify get_trainer returns the MLX trainer identifier."""
    assert get_trainer() == "mlx_trainer"


def test_all_exports() -> None:
    """Verify __all__ exports all required MLX backend entry points."""
    expected_exports = ["apply_lora", "benchmark_model", "build_dataloader", "export_model", "generate_sql", "get_trainer", "log_metrics", "quantize_model", "run_dpo", "serve_model", "train_model"]
    for exp in expected_exports:
        assert exp in __all__


def test_mlx_logging_missing_and_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify MLX log_metrics handles missing and present SummaryWriter."""
    import gemma_4_sql.backends.mlx.logging as m_log

    monkeypatch.setattr(m_log, "SummaryWriter", None)
    res_missing = m_log.log_metrics({"loss": 1.0}, 1, str(tmp_path / "test"))
    assert "backend_unavailable" in str(res_missing["status"])

    monkeypatch.setattr(m_log, "SummaryWriter", MagicMock())
    res_ok = m_log.log_metrics({"loss": 1.0}, 1, str(tmp_path / "test"))
    assert res_ok["status"] == "success"
