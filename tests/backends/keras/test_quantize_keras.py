"""Tests for Keras quantization logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gemma_4_sql.backends.keras.quantize as kr_quantize
from gemma_4_sql.backends.keras.quantize import quantize_model

if TYPE_CHECKING:
    import pytest


def test_quantize_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras quantize when missing."""
    monkeypatch.setattr(kr_quantize, "keras", None)
    res = quantize_model("model", "int8")
    if not res["status"] == "mocked_missing_keras":
        raise AssertionError
    if not res["memory_reduction_factor"] == 0.0:
        raise AssertionError


def test_quantize_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras quantize."""
    monkeypatch.setattr(kr_quantize, "keras", object())

    res = quantize_model("model", "int8")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["status"] == "quantized_int8":
        raise AssertionError

    res = quantize_model("model", "int4")
    if not res["status"] == "quantized_int4":
        raise AssertionError

    res = quantize_model("model", "awq")
    if not res["status"] == "quantized_awq":
        raise AssertionError

    res = quantize_model("model", "unknown")
    if "unsupported" not in str(res["status"]):
        raise AssertionError
