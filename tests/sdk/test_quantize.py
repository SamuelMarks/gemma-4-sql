"""Tests for Quantization SDK module."""

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.quantize import quantize_model


def test_quantize_jax() -> None:
    """Initialize function test_quantize_jax.

    Raises:
        AssertionError: Description.

    """
    res = quantize_model("model1", "int8", backend="jax")
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["method"] == "int8":
        raise AssertionError


def test_quantize_pytorch() -> None:
    """Initialize function test_quantize_pytorch.

    Raises:
        AssertionError: Description.

    """
    with pytest.raises(DependencyMissingError):
        quantize_model("model2", "int4", backend="pytorch")


def test_quantize_keras() -> None:
    """Initialize function test_quantize_keras.

    Raises:
        AssertionError: Description.

    """
    res = quantize_model("model3", "awq", backend="keras")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["model"] == "model3":
        raise AssertionError


def test_quantize_maxtext() -> None:
    """Initialize function test_quantize_maxtext."""
    with pytest.raises(DependencyMissingError):
        quantize_model("model4", "gguf", backend="maxtext")


def test_quantize_invalid() -> None:
    """Initialize function test_quantize_invalid."""
    with pytest.raises(ValueError, match=r".*"):
        quantize_model("model", "int8", backend="unknown")
