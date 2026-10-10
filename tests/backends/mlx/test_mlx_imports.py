"""Module docstring."""

import sys
from unittest.mock import MagicMock


def test_mlx_imports():
    """Docstring for test_mlx_imports."""
    sys.modules["mlx"] = MagicMock()
    sys.modules["mlx.core"] = MagicMock()
    sys.modules["mlx.nn"] = MagicMock()
    sys.modules["mlx.optimizers"] = MagicMock()
    sys.modules["mlx.utils"] = MagicMock()

    import gemma_4_sql.backends.mlx as mlx_init

    assert mlx_init.get_trainer() == "mlx_trainer"


def test_mlx_logging_missing_tb(monkeypatch):
    """Docstring for test_mlx_logging_missing_tb."""
    import gemma_4_sql.backends.mlx.logging as m_log

    monkeypatch.setattr(m_log, "SummaryWriter", None)
    res = m_log.log_metrics({"loss": 1.0}, 1, "test")
    assert "mocked" in res["status"]


def test_mlx_logging_success(monkeypatch):
    """Docstring for test_mlx_logging_success."""
    import gemma_4_sql.backends.mlx.logging as m_log

    mock_sw = MagicMock()
    monkeypatch.setattr(m_log, "SummaryWriter", mock_sw)
    res = m_log.log_metrics({"loss": 1.0}, 1, "test")
    assert res["status"] == "success"
