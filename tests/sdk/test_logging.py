# Copyright 2024
"""Tests for SDK Logging module."""

import pytest

from gemma_4_sql.sdk.logging import log_metrics


def test_log_metrics_jax() -> None:
    """Test logging with jax.

    Raises:
        AssertionError: Description.

    """
    res = log_metrics({"loss": 0.5}, 10, backend="jax")
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["metrics"] == {"loss": 0.5}:
        raise AssertionError
    if not res["step"] == int("10"):
        raise AssertionError


def test_log_metrics_keras() -> None:
    """Test logging with keras.

    Raises:
        AssertionError: Description.

    """
    res = log_metrics({"loss": 0.5}, 10, backend="keras")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["metrics"] == {"loss": 0.5}:
        raise AssertionError
    if not res["step"] == int("10"):
        raise AssertionError


def test_log_metrics_maxtext() -> None:
    """Test logging with maxtext.

    Raises:
        AssertionError: Description.

    """
    res = log_metrics({"loss": 0.5}, 10, backend="maxtext")
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["metrics"] == {"loss": 0.5}:
        raise AssertionError
    if not res["step"] == int("10"):
        raise AssertionError


def test_log_metrics_pytorch() -> None:
    """Test logging with pytorch.

    Raises:
        AssertionError: Description.

    """
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
