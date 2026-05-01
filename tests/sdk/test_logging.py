"""
Tests for SDK Logging module.
"""

import pytest

from gemma_4_sql.sdk.logging import log_metrics


def test_log_metrics_jax() -> None:
    """Test logging with jax."""
    res = log_metrics({"loss": 0.5}, 10, backend="jax")
    assert res["backend"] == "jax"
    assert res["metrics"] == {"loss": 0.5}
    assert res["step"] == 10
    # Will be mocked missing tb in test environment by default
    assert "mocked" in res["status"]


def test_log_metrics_keras() -> None:
    """Test logging with keras."""
    res = log_metrics({"loss": 0.5}, 10, backend="keras")
    assert res["backend"] == "keras"
    assert res["metrics"] == {"loss": 0.5}
    assert res["step"] == 10
    assert "mocked" in res["status"]


def test_log_metrics_maxtext() -> None:
    """Test logging with maxtext."""
    res = log_metrics({"loss": 0.5}, 10, backend="maxtext")
    assert res["backend"] == "maxtext"
    assert res["metrics"] == {"loss": 0.5}
    assert res["step"] == 10
    assert "mocked" in res["status"]


def test_log_metrics_pytorch() -> None:
    """Test logging with pytorch."""
    res = log_metrics({"loss": 0.5}, 10, backend="pytorch")
    assert res["backend"] == "pytorch"
    assert res["metrics"] == {"loss": 0.5}
    assert res["step"] == 10
    assert "mocked" in res["status"]


def test_log_metrics_invalid() -> None:
    """Test logging with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        log_metrics({"loss": 0.5}, 10, backend="invalid")
