"""Unit and behavioral tests for logging keras."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.keras.logging as log_keras
from gemma_4_sql.backends.keras.logging import log_metrics
from tests.conftest import requires_keras

pytestmark = requires_keras


def test_log_metrics_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify Keras log_metrics creates a TensorFlow summary writer and records scalar metrics."""
    mock_tf = MagicMock()
    mock_writer = MagicMock()
    mock_tf.summary.create_file_writer.return_value = mock_writer
    monkeypatch.setattr(log_keras, "tf", mock_tf)

    log_dir = str(tmp_path / "keras_logs")
    res = log_metrics({"loss": 0.5, "acc": 0.9}, 5, log_dir)
    assert res["status"] == "success"
    assert res["backend"] == "keras"
    assert res["action"] == "log_metrics"
    assert res["step"] == 5
    mock_tf.summary.create_file_writer.assert_called_once_with(log_dir)
    assert mock_tf.summary.scalar.call_count == 2
    mock_writer.close.assert_called_once()


def test_log_metrics_missing_tf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Keras log_metrics raises DependencyMissingError when tf is None."""
    monkeypatch.setattr(log_keras, "tf", None)
    with pytest.raises(Exception, match="TensorFlow dependencies are missing"):
        log_metrics({"loss": 0.5}, 5, "logs")


def test_log_metrics_missing_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify Keras log_metrics handles tf object lacking summary attribute."""

    class DummyTF:
        """Stub TensorFlow module without summary attribute."""

    monkeypatch.setattr(log_keras, "tf", DummyTF())
    monkeypatch.setattr("gemma_4_sql.backends.common_logging.get_tensorboard_summary_writer", lambda: None)
    res = log_metrics({"loss": 0.5}, 5, str(tmp_path / "no_summary"), allow_jsonl_fallback=True)
    assert res["status"] == "missing_summary_attr"


def test_log_metrics_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Keras log_metrics propagates RuntimeError when summary writer creation fails."""
    mock_tf: Any = MagicMock()
    mock_tf.summary.create_file_writer.side_effect = RuntimeError("TF error")
    monkeypatch.setattr(log_keras, "tf", mock_tf)

    with pytest.raises(RuntimeError, match="TF error"):
        log_metrics({"loss": 0.5}, 5, "logs")
