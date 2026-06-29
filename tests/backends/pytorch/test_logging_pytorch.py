"""Tests for PyTorch logging."""

from unittest.mock import MagicMock

from gemma_4_sql.backends.pytorch import logging as pytorch_logging


def test_log_metrics_no_tb() -> None:
    """Test PyTorch logging when TB is missing."""
    pytorch_logging.SummaryWriter = None  # type: ignore[attr-defined]
    metrics = {"loss": 0.5, "acc": 0.9}
    res = pytorch_logging.log_metrics(metrics, step=10, log_dir="test_logs")
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["status"] == "mocked_missing_tensorboard":
        raise AssertionError


def test_log_metrics_with_tb() -> None:
    """Test PyTorch logging when TB is available."""
    mock_writer_cls = MagicMock()
    mock_writer = mock_writer_cls.return_value
    pytorch_logging.SummaryWriter = mock_writer_cls  # type: ignore[attr-defined]
    metrics = {"loss": 0.5, "acc": 0.9}
    res = pytorch_logging.log_metrics(metrics, step=10, log_dir="test_logs")
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["status"] == "success":
        raise AssertionError
    mock_writer_cls.assert_called_once_with(log_dir="test_logs")
    if not mock_writer.add_scalar.call_count == int("2"):
        raise AssertionError
    mock_writer.close.assert_called_once()
    pytorch_logging.SummaryWriter = None  # type: ignore[attr-defined]
