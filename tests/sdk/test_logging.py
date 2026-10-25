"""Tests for Logging SDK module."""

import importlib.util
import json
from pathlib import Path

import pytest

from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError
from gemma_4_sql.sdk.logging import log_metrics

has_tf = importlib.util.find_spec("tensorflow") is not None


def test_log_metrics_jax(tmp_path: Path) -> None:
    """Verify log metrics jax behavior and JSONL fallback file persistence."""
    log_dir = str(tmp_path / "jax_logs")
    res = log_metrics({"loss": 0.5}, 10, log_dir=log_dir, backend="jax")
    assert res["backend"] == "jax"
    assert res.get("action") == "log_metrics"
    assert res.get("step") == 10
    assert res.get("metrics") == {"loss": 0.5}
    fallback_file = res.get("fallback_file")
    assert fallback_file is not None
    lines = Path(fallback_file).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["step"] == 10
    assert record["metrics"] == {"loss": 0.5}


def test_log_metrics_pytorch(tmp_path: Path) -> None:
    """Verify log metrics pytorch behavior."""
    res = log_metrics({"loss": 0.5}, 10, log_dir=str(tmp_path / "pt_logs"), backend="pytorch")
    assert res["backend"] == "pytorch"
    assert res.get("metrics") == {"loss": 0.5}


def test_log_metrics_keras(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify log metrics keras behavior."""
    from unittest.mock import MagicMock

    import gemma_4_sql.backends.keras.logging as klog

    mock_tf = MagicMock()
    monkeypatch.setattr(klog, "tf", mock_tf)
    res = log_metrics({"loss": 0.5}, 10, log_dir=str(tmp_path / "keras_logs"), backend="keras")
    assert res["backend"] == "keras"
    assert res.get("metrics") == {"loss": 0.5}


def test_log_metrics_maxtext() -> None:
    """Test logging with maxtext."""
    with pytest.raises(DependencyMissingError):
        log_metrics({"loss": 0.5}, 10, backend="maxtext")


def test_log_metrics_invalid() -> None:
    """Verify log_metrics raises ConfigurationError for an unknown backend."""
    with pytest.raises(ConfigurationError, match="Unknown backend: unknown"):
        log_metrics({"loss": 0.5}, 10, backend="unknown")
