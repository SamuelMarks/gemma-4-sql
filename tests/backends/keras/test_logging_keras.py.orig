"""Tests for Keras logging."""

from unittest.mock import MagicMock

from gemma_4_sql.backends.keras import logging as keras_logging


def test_log_metrics_no_tb() -> None:
    """Test Keras logging when TB is missing."""
    keras_logging.tf = None  # type: ignore[attr-defined]
    metrics = {"loss": 0.5, "acc": 0.9}
    res = keras_logging.log_metrics(metrics, step=10, log_dir="test_logs")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["status"] == "mocked_missing_tensorboard":
        raise AssertionError


def test_log_metrics_with_tb() -> None:
    """Test Keras logging when TB is available."""
    mock_tf = MagicMock()
    mock_writer = MagicMock()
    mock_tf.summary.create_file_writer.return_value = mock_writer
    keras_logging.tf = mock_tf  # type: ignore[attr-defined]
    metrics = {"loss": 0.5, "acc": 0.9}
    res = keras_logging.log_metrics(metrics, step=10, log_dir="test_logs")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["status"] == "success":
        raise AssertionError
    mock_tf.summary.create_file_writer.assert_called_once_with("test_logs")
    if not mock_tf.summary.scalar.call_count == int("2"):
        raise AssertionError
    mock_writer.close.assert_called_once()
    keras_logging.tf = None  # type: ignore[attr-defined]
