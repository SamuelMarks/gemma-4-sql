"""
Tests for MaxText quantization logic.
"""

from __future__ import annotations

import pytest

from gemma_4_sql.backends.maxtext.quantize import quantize_model


def test_quantize_maxtext_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MaxText quantize when missing."""
    import gemma_4_sql.backends.maxtext.quantize as maxtext_quantize

    monkeypatch.setattr(maxtext_quantize, "jnp", None)

    res = quantize_model("model", "int8")
    assert res["status"] == "mocked_missing_maxtext"
    assert res["memory_reduction_factor"] == 0.0


def test_quantize_maxtext() -> None:
    """Test MaxText quantize."""
    res = quantize_model("model", "awq")
    assert res["backend"] == "maxtext"
    assert res["method"] == "awq"
    assert res["status"] in ["quantized_awq", "mocked_missing_maxtext"]
