"""Module docstring."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.maxtext.logging import log_metrics
from gemma_4_sql.exceptions import DependencyMissingError


def test_log_metrics_success():
    """Docstring for test_log_metrics_success."""
    metrics = {"loss": 0.5, "accuracy": 0.9}
    step = 10
    log_dir = "test_logs"

    mock_writer_instance = MagicMock()
    mock_SummaryWriter = MagicMock(return_value=mock_writer_instance)

    with patch("gemma_4_sql.backends.maxtext.logging.SummaryWriter", new=mock_SummaryWriter):
        result = log_metrics(metrics, step, log_dir)

    assert result == {
        "backend": "maxtext",
        "action": "log_metrics",
        "step": step,
        "metrics": metrics,
        "status": "success",
        "log_dir": log_dir,
    }

    mock_SummaryWriter.assert_called_once_with(log_dir=log_dir)
    assert mock_writer_instance.add_scalar.call_count == 2
    mock_writer_instance.add_scalar.assert_any_call("loss", 0.5, step)
    mock_writer_instance.add_scalar.assert_any_call("accuracy", 0.9, step)
    mock_writer_instance.close.assert_called_once()


def test_log_metrics_missing_dependency():
    """Docstring for test_log_metrics_missing_dependency."""
    with patch("gemma_4_sql.backends.maxtext.logging.SummaryWriter", new=None), pytest.raises(DependencyMissingError, match="TensorBoardX dependencies are missing."):
        log_metrics({"loss": 0.5}, 10)


def test_module_load_import_error(monkeypatch):
    """Test the try/except ImportError block."""
    import builtins
    import sys

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "tensorboardX":
            raise ImportError("simulated missing import")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        if "gemma_4_sql.backends.maxtext.logging" in sys.modules:
            del sys.modules["gemma_4_sql.backends.maxtext.logging"]
        import gemma_4_sql.backends.maxtext.logging as mlog

        assert mlog.SummaryWriter is None

    # Restore
    if "gemma_4_sql.backends.maxtext.logging" in sys.modules:
        del sys.modules["gemma_4_sql.backends.maxtext.logging"]


def test_tensorboardx_import_success(monkeypatch):
    """Docstring for test_tensorboardx_import_success."""
    import importlib
    import sys
    from unittest.mock import MagicMock

    mock_tb = MagicMock()
    mock_tb.SummaryWriter = "MockWriter"
    monkeypatch.setitem(sys.modules, "tensorboardX", mock_tb)

    import gemma_4_sql.backends.maxtext.logging as log_max

    importlib.reload(log_max)

    assert log_max.SummaryWriter == "MockWriter"

    monkeypatch.undo()
    importlib.reload(log_max)
