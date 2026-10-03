"""Tests for Keras logging."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.keras.logging import log_metrics
from gemma_4_sql.exceptions import DependencyMissingError


def test_log_metrics_missing_tf():
    with patch("gemma_4_sql.backends.keras.logging.tf", None), pytest.raises(DependencyMissingError, match="TensorFlow dependencies are missing"):
        log_metrics({"a": 1}, 1)


def test_log_metrics_missing_summary_attr():
    mock_tf = MagicMock()
    del mock_tf.summary
    with patch("gemma_4_sql.backends.keras.logging.tf", mock_tf):
        res = log_metrics({"a": 1}, 1)
        assert res["status"] == "missing_summary_attr"


def test_log_metrics_success():
    mock_tf = MagicMock()
    mock_writer = MagicMock()
    mock_tf.summary.create_file_writer.return_value = mock_writer

    with patch("gemma_4_sql.backends.keras.logging.tf", mock_tf):
        res = log_metrics({"a": 1.0, "b": 2.0}, 5, "logs")
        assert res["status"] == "success"
        assert res["step"] == 5
        mock_writer.close.assert_called_once()
