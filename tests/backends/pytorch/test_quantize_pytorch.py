"""
Tests for PyTorch quantization logic.
"""

from __future__ import annotations

import pytest

from gemma_4_sql.backends.pytorch.quantize import quantize_model


def test_quantize_pytorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch quantize when missing."""
    import gemma_4_sql.backends.pytorch.quantize as torch_quantize

    monkeypatch.setattr(torch_quantize, "torch", None)

    res = quantize_model("model", "int8")
    assert res["status"] == "mocked_missing_torch"
    assert res["memory_reduction_factor"] == 0.0


def test_quantize_pytorch() -> None:
    """Test PyTorch quantize."""
    res = quantize_model("model", "awq")
    assert res["backend"] == "pytorch"
    assert res["method"] == "awq"
    assert res["status"] in ["quantized_awq", "mocked_missing_torch"]
