"""Tests for SDK quantize module."""

from __future__ import annotations

import pytest
from gemma_4_sql.sdk.quantize import quantize_model


def test_quantize_pytorch() -> None:
    """Initialize function test_quantize_pytorch."""
    res = quantize_model("model1", "int8", backend="pytorch")
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["method"] == "int8":
        raise AssertionError


def test_quantize_jax() -> None:
    """Initialize function test_quantize_jax."""
    res = quantize_model("model2", "awq", backend="jax")
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["model"] == "model2":
        raise AssertionError
    if not res["method"] == "awq":
        raise AssertionError


def test_quantize_keras() -> None:
    """Initialize function test_quantize_keras."""
    res = quantize_model("model3", "gptq", backend="keras")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["model"] == "model3":
        raise AssertionError
    if not res["method"] == "gptq":
        raise AssertionError


def test_quantize_maxtext() -> None:
    """Initialize function test_quantize_maxtext."""
    res = quantize_model("model4", "gguf", backend="maxtext")
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["model"] == "model4":
        raise AssertionError
    if not res["method"] == "gguf":
        raise AssertionError


def test_quantize_unknown_backend() -> None:
    """Initialize function test_quantize_unknown_backend."""
    with pytest.raises(ValueError, match="Unknown backend: missing"):
        quantize_model("my-model", "int8", backend="missing")
