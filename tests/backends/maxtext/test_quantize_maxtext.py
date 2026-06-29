"""Tests for MaxText quantization logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.backends.maxtext.quantize import quantize_model

if TYPE_CHECKING:
    import pytest


def test_quantize_maxtext_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MaxText quantize when missing."""
    maxtext_quantize = __import__("gemma_4_sql.backends.maxtext.quantize", fromlist=[""])
    monkeypatch.setattr(maxtext_quantize, "jnp", None)
    res = quantize_model("model", "int8")
    if not res["status"] == "mocked_missing_maxtext":
        raise AssertionError
    if not res["memory_reduction_factor"] == 0.0:
        raise AssertionError


def test_quantize_maxtext() -> None:
    """Test MaxText quantize."""
    res = quantize_model("model", "awq")
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["method"] == "awq":
        raise AssertionError
    if res["status"] not in ["quantized_awq", "mocked_missing_maxtext"]:
        raise AssertionError
