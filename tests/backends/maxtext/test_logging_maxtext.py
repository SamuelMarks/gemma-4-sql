"""Unit and behavioral tests for logging maxtext."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.maxtext.logging import log_metrics
from gemma_4_sql.exceptions import DependencyMissingError


def test_log_metrics_success():
    """Verify log metrics success behavior."""
    metrics = {"loss": 0.5, "accuracy": 0.9}
    step = 10
    log_dir = "test_logs"

    mock_writer_instance = MagicMock()
    mock_SummaryWriter = MagicMock(return_value=mock_writer_instance)

    with patch("gemma_4_sql.backends.maxtext.logging.SummaryWriter", new=mock_SummaryWriter):
        result = log_metrics(metrics, step, log_dir)

    assert result["backend"] == "maxtext"
    assert result.get("action") == "log_metrics"
    assert result.get("step") == step
    assert result.get("metrics") == metrics
    assert result.get("logged_metrics") == metrics
    assert result["status"] == "success"
    assert result["log_dir"] == log_dir
    assert "timestamp" in result

    mock_SummaryWriter.assert_called_once_with(log_dir=log_dir)
    assert mock_writer_instance.add_scalar.call_count == 2
    mock_writer_instance.add_scalar.assert_any_call("loss", 0.5, step)
    mock_writer_instance.add_scalar.assert_any_call("accuracy", 0.9, step)
    mock_writer_instance.close.assert_called_once()


def test_log_metrics_missing_dependency(tmp_path: Path):
    """Verify log metrics missing dependency behavior."""
    with patch("gemma_4_sql.backends.maxtext.logging.SummaryWriter", new=None):
        with pytest.raises(DependencyMissingError, match=r"TensorBoardX dependencies are missing\."):
            log_metrics({"loss": 0.5}, 10)
        fallback_res = log_metrics({"loss": 0.5}, 10, log_dir=str(tmp_path), allow_jsonl_fallback=True)
        assert fallback_res["status"] == "backend_unavailable"
        assert "fallback_file" in fallback_res
