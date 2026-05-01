"""Tests for JAX logging."""

from unittest.mock import MagicMock

from gemma_4_sql.backends.jax import logging as jax_logging


def test_log_metrics_no_tb() -> None:
    """Test JAX logging when TB is missing."""
    jax_logging.SummaryWriter = None
    metrics = {"loss": 0.5, "acc": 0.9}
    res = jax_logging.log_metrics(metrics, step=10, log_dir="test_logs")
    assert res["backend"] == "jax"
    assert res["status"] == "mocked_missing_tensorboard"


def test_log_metrics_with_tb() -> None:
    """Test JAX logging when TB is available."""
    mock_writer_cls = MagicMock()
    mock_writer = mock_writer_cls.return_value
    jax_logging.SummaryWriter = mock_writer_cls

    metrics = {"loss": 0.5, "acc": 0.9}
    res = jax_logging.log_metrics(metrics, step=10, log_dir="test_logs")

    assert res["backend"] == "jax"
    assert res["status"] == "success"
    mock_writer_cls.assert_called_once_with(log_dir="test_logs")
    assert mock_writer.add_scalar.call_count == 2
    mock_writer.close.assert_called_once()

    jax_logging.SummaryWriter = None  # Reset
