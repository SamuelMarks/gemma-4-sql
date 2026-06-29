"""Tests for JAX quantization logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.backends.jax.quantize import quantize_model

if TYPE_CHECKING:
    import pytest


def test_quantize_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX quantize when missing."""
    jax_quantize = __import__("gemma_4_sql.backends.jax.quantize", fromlist=[""])
    monkeypatch.setattr(jax_quantize, "jax", None)
    res = quantize_model("model", "int8")
    if not res["status"] == "mocked_missing_jax":
        raise AssertionError
    if not res["memory_reduction_factor"] == 0.0:
        raise AssertionError


def test_quantize_jax() -> None:
    """Test JAX quantize."""
    res = quantize_model("model", "awq")
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["method"] == "awq":
        raise AssertionError
    if res["status"] not in ["quantized_awq", "mocked_missing_jax"]:
        raise AssertionError
