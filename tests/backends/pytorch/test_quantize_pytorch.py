"""Tests for PyTorch quantization logic."""

from __future__ import annotations

import pytest

import gemma_4_sql.backends.pytorch.quantize as pt_quantize
from gemma_4_sql.backends.pytorch.quantize import quantize_model


class MockTorch:
    """Provide class docstring."""

    float16 = "float16"


class MockBitsAndBytesConfig:
    """Provide class docstring."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Execute function."""


class MockAutoModelForCausalLM:
    """Provide class docstring."""

    @staticmethod
    def from_pretrained(_model_name: str, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return object()


def test_quantize_pytorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch quantize when missing.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(pt_quantize, "torch", None)
    monkeypatch.setattr(pt_quantize, "BitsAndBytesConfig", None)
    monkeypatch.setattr(pt_quantize, "AutoModelForCausalLM", None)
    with pytest.raises(DependencyMissingError, match="PyTorch quantization dependencies are missing."):
        quantize_model("model", "int8")


def test_quantize_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch quantize.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt_quantize, "torch", MockTorch())
    monkeypatch.setattr(pt_quantize, "BitsAndBytesConfig", MockBitsAndBytesConfig)
    monkeypatch.setattr(pt_quantize, "AutoModelForCausalLM", MockAutoModelForCausalLM)
    res = quantize_model("model", "int8")
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["status"] == "quantized_int8":
        raise AssertionError
    res = quantize_model("model", "int4")
    if not res["status"] == "quantized_int4":
        raise AssertionError
    res = quantize_model("model", "awq")
    if res["status"] not in {"quantized_awq", "mocked_missing_torch"}:
        raise AssertionError
    res = quantize_model("model", "unknown")
    if "unsupported" not in res["status"]:
        raise AssertionError


def test_quantize_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt_quantize, "torch", MockTorch())
    monkeypatch.setattr(pt_quantize, "BitsAndBytesConfig", MockBitsAndBytesConfig)
    monkeypatch.setattr(pt_quantize, "AutoModelForCausalLM", MockAutoModelForCausalLM)

    def mock_raise_error(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(pt_quantize, "BitsAndBytesConfig", Exception)
    res = quantize_model("model", "int8")
    if "failed" not in str(res["status"]):
        raise AssertionError
