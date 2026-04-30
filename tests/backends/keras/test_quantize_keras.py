"""
Tests for Keras quantization logic.
"""

from __future__ import annotations

import pytest

from gemma_4_sql.backends.keras.quantize import quantize_model


def test_quantize_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras quantize when missing."""
    import gemma_4_sql.backends.keras.quantize as keras_quantize

    monkeypatch.setattr(keras_quantize, "tf", None)

    res = quantize_model("model", "int8")
    assert res["status"] == "mocked_missing_keras"
    assert res["memory_reduction_factor"] == 0.0


def test_quantize_keras() -> None:
    """Test Keras quantize."""
    res = quantize_model("model", "awq")
    assert res["backend"] == "keras"
    assert res["method"] == "awq"
    assert res["status"] in ["quantized_awq", "mocked_missing_keras"]
