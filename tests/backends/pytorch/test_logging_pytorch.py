"""Tests for PyTorch logging."""

from unittest.mock import patch

from tests.conftest import requires_torch

pytestmark = requires_torch


def test_pytorch_logging_imports(monkeypatch):
    """Test pytorch logging imports fallback."""
    import gemma_4_sql.backends.pytorch.logging as logging_module

    monkeypatch.setattr(logging_module, "SummaryWriter", None)
    assert logging_module.SummaryWriter is None


def test_log_metrics():
    """Test log_metrics."""
    import gemma_4_sql.backends.pytorch.logging as logging_module

    with patch("gemma_4_sql.backends.pytorch.logging.log_metrics_wrapper") as mock_wrapper:
        mock_wrapper.return_value = {"status": "ok"}

        res = logging_module.log_metrics({"m": 1.0}, 1, "dir")

        assert res == {"status": "ok"}
        mock_wrapper.assert_called_once_with(backend_name="pytorch", metrics={"m": 1.0}, step=1, log_dir="dir", summary_writer_cls=logging_module.SummaryWriter)
