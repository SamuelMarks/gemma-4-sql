"""Tests for SDK Logging module."""

import pytest
from gemma_4_sql.sdk.logging import log_metrics


def test_log_metrics_jax() -> None:
    """Test logging with jax."""
    res = log_metrics({"loss": 0.5}, 10, backend="jax")
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["metrics"] == {"loss": 0.5}:
        raise AssertionError
    if not res["step"] == int("10"):
        raise AssertionError


def test_log_metrics_keras() -> None:
    """Test logging with keras."""
    res = log_metrics({"loss": 0.5}, 10, backend="keras")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["metrics"] == {"loss": 0.5}:
        raise AssertionError
    if not res["step"] == int("10"):
        raise AssertionError


def test_log_metrics_maxtext() -> None:
    """Test logging with maxtext."""
    res = log_metrics({"loss": 0.5}, 10, backend="maxtext")
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["metrics"] == {"loss": 0.5}:
        raise AssertionError
    if not res["step"] == int("10"):
        raise AssertionError


def test_log_metrics_pytorch() -> None:
    """Test logging with pytorch."""
    res = log_metrics({"loss": 0.5}, 10, backend="pytorch")
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["metrics"] == {"loss": 0.5}:
        raise AssertionError
    if not res["step"] == int("10"):
        raise AssertionError


def test_log_metrics_invalid() -> None:
    """Test logging with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        log_metrics({"loss": 0.5}, 10, backend="invalid")
