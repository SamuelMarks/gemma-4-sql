"""
Tests for JAX quantization logic.
"""

from __future__ import annotations

import pytest

from gemma_4_sql.backends.jax.quantize import quantize_model


def test_quantize_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX quantize when missing."""
    import gemma_4_sql.backends.jax.quantize as jax_quantize

    monkeypatch.setattr(jax_quantize, "jax", None)

    res = quantize_model("model", "int8")
    assert res["status"] == "mocked_missing_jax"
    assert res["memory_reduction_factor"] == 0.0


def test_quantize_jax() -> None:
    """Test JAX quantize."""
    res = quantize_model("model", "awq")
    assert res["backend"] == "jax"
    assert res["method"] == "awq"
    assert res["status"] in ["quantized_awq", "mocked_missing_jax"]
