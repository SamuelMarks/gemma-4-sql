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
