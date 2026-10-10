"""Tests for PyTorch logging."""

import sys
from unittest.mock import patch


def test_pytorch_logging_imports():
    """Test pytorch logging imports fallback."""
    import importlib

    with patch.dict(sys.modules, {"torch": None, "torch.utils": None, "torch.utils.tensorboard": None}):
        import gemma_4_sql.backends.pytorch.logging as logging_module

        importlib.reload(logging_module)
        assert logging_module.SummaryWriter is None
    importlib.reload(logging_module)


def test_log_metrics():
    """Test log_metrics."""
    import gemma_4_sql.backends.pytorch.logging as logging_module

    with patch("gemma_4_sql.backends.pytorch.logging.log_metrics_wrapper") as mock_wrapper:
        mock_wrapper.return_value = {"status": "ok"}

        res = logging_module.log_metrics({"m": 1.0}, 1, "dir")

        assert res == {"status": "ok"}
        mock_wrapper.assert_called_once_with(backend_name="pytorch", metrics={"m": 1.0}, step=1, log_dir="dir", summary_writer_cls=logging_module.SummaryWriter)
