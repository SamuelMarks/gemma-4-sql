"""Tests for PyTorch quantization logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.backends.pytorch.quantize import quantize_model

if TYPE_CHECKING:
    import pytest


def test_quantize_pytorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch quantize when missing."""
    torch_quantize = __import__("gemma_4_sql.backends.pytorch.quantize", fromlist=[""])
    monkeypatch.setattr(torch_quantize, "torch", None)
    res = quantize_model("model", "int8")
    if not res["status"] == "mocked_missing_torch":
        raise AssertionError
    if not res["memory_reduction_factor"] == 0.0:
        raise AssertionError


def test_quantize_pytorch() -> None:
    """Test PyTorch quantize."""
    res = quantize_model("model", "awq")
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["method"] == "awq":
        raise AssertionError
    if res["status"] not in ["quantized_awq", "mocked_missing_torch"]:
        raise AssertionError
