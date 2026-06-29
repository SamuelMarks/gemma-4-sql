"""Tests for Keras quantization logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.backends.keras.quantize import quantize_model

if TYPE_CHECKING:
    import pytest


def test_quantize_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras quantize when missing."""
    keras_quantize = __import__("gemma_4_sql.backends.keras.quantize", fromlist=[""])
    monkeypatch.setattr(keras_quantize, "tf", None)
    res = quantize_model("model", "int8")
    if not res["status"] == "mocked_missing_keras":
        raise AssertionError
    if not res["memory_reduction_factor"] == 0.0:
        raise AssertionError


def test_quantize_keras() -> None:
    """Test Keras quantize."""
    res = quantize_model("model", "awq")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["method"] == "awq":
        raise AssertionError
    if res["status"] not in ["quantized_awq", "mocked_missing_keras"]:
        raise AssertionError
