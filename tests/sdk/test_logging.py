"""Tests for Logging SDK module."""

import importlib.util

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.logging import log_metrics

has_tf = importlib.util.find_spec("tensorflow") is not None


def test_log_metrics_jax() -> None:
    """Verify log metrics jax behavior."""
    res = log_metrics({"loss": 0.5}, 10, backend="jax")
    assert res["backend"] == "jax"

    assert res.get("action") == "log_metrics"

    assert res.get("step") == 10

    assert res.get("metrics") == {"loss": 0.5}


def test_log_metrics_pytorch() -> None:
    """Verify log metrics pytorch behavior."""
    res = log_metrics({"loss": 0.5}, 10, backend="pytorch")
    assert res["backend"] == "pytorch"

    assert res.get("metrics") == {"loss": 0.5}


@pytest.mark.skipif(not has_tf, reason="TensorFlow is not installed")
def test_log_metrics_keras() -> None:
    """Verify log metrics keras behavior."""
    res = log_metrics({"loss": 0.5}, 10, backend="keras")
    assert res["backend"] == "keras"

    assert res.get("metrics") == {"loss": 0.5}


def test_log_metrics_maxtext() -> None:
    """Test logging with maxtext."""
    with pytest.raises(DependencyMissingError):
        log_metrics({"loss": 0.5}, 10, backend="maxtext")


def test_log_metrics_invalid() -> None:
    """Test logging with invalid."""
    with pytest.raises(ValueError, match=r".*"):
        log_metrics({"loss": 0.5}, 10, backend="unknown")
