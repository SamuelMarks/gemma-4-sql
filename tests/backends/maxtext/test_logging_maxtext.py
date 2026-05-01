"""Tests for MaxText logging."""

from unittest.mock import MagicMock

from gemma_4_sql.backends.maxtext import logging as maxtext_logging


def test_log_metrics_no_tb() -> None:
    """Test MaxText logging when TB is missing."""
    maxtext_logging.SummaryWriter = None
    metrics = {"loss": 0.5, "acc": 0.9}
    res = maxtext_logging.log_metrics(metrics, step=10, log_dir="test_logs")
    assert res["backend"] == "maxtext"
    assert res["status"] == "mocked_missing_tensorboard"


def test_log_metrics_with_tb() -> None:
    """Test MaxText logging when TB is available."""
    mock_writer_cls = MagicMock()
    mock_writer = mock_writer_cls.return_value
    maxtext_logging.SummaryWriter = mock_writer_cls

    metrics = {"loss": 0.5, "acc": 0.9}
    res = maxtext_logging.log_metrics(metrics, step=10, log_dir="test_logs")

    assert res["backend"] == "maxtext"
    assert res["status"] == "success"
    mock_writer_cls.assert_called_once_with(log_dir="test_logs")
    assert mock_writer.add_scalar.call_count == 2
    mock_writer.close.assert_called_once()

    maxtext_logging.SummaryWriter = None  # Reset
