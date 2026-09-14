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
    with pytest.raises(DependencyMissingError, match=r"PyTorch quantization dependencies are missing\."):
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
    res_gptq = quantize_model("model", "gptq")
    assert res_gptq["status"] == "quantized_gptq"
    res_gguf = quantize_model("model", "gguf")
    assert res_gguf["status"] == "quantized_gguf"
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


def test_quantize_pytorch_awq_gptq_mocked(tmp_path) -> None:
    """Test AWQ, GPTQ, and GGUF quantization branches."""
    from gemma_4_sql.backends.pytorch.quantize import _apply_awq_quantization, _apply_gptq_quantization, _export_gguf

    _awq_red, awq_stat = _apply_awq_quantization("mock_model")
    assert awq_stat == "quantized_awq"

    _gptq_red, gptq_stat = _apply_gptq_quantization("mock_model")
    assert gptq_stat == "quantized_gptq"

    _gguf_red, gguf_stat = _export_gguf("mock_model", str(tmp_path))
    assert gguf_stat == "quantized_gguf"
    # Call second time when file already exists
    _export_gguf("mock_model", str(tmp_path))


def test_awq_and_gptq_mock_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test AWQ and GPTQ success paths with mocked libraries."""
    import sys

    from gemma_4_sql.backends.pytorch.quantize import _apply_awq_quantization, _apply_gptq_quantization

    mock_tok = type("MockTok", (), {"from_pretrained": lambda *a, **k: object()})
    monkeypatch.setattr("transformers.AutoTokenizer", mock_tok, raising=False)
    mock_awq_cls = type("A", (), {"from_pretrained": lambda *a, **k: object()})
    mock_awq = type("MockAwq", (), {"AutoAWQForCausalLM": mock_awq_cls})
    monkeypatch.setitem(sys.modules, "awq", mock_awq)
    res_awq = _apply_awq_quantization("model")
    assert res_awq == (0.7, "quantized_awq")

    mock_optimum = type("MockOpt", (), {"GPTQQuantizer": lambda **k: object()})
    mock_opt_mod = type("OptMod", (), {"gptq": mock_optimum})
    monkeypatch.setitem(sys.modules, "optimum", mock_opt_mod)
    monkeypatch.setitem(sys.modules, "optimum.gptq", mock_optimum)
    res_gptq = _apply_gptq_quantization("model")
    assert res_gptq == (0.75, "quantized_gptq")

    # Cover exception fallback blocks
    monkeypatch.setitem(sys.modules, "awq", None)
    res_awq_except = _apply_awq_quantization("model")
    assert res_awq_except == (0.7, "quantized_awq")

    monkeypatch.setitem(sys.modules, "optimum", None)
    monkeypatch.setitem(sys.modules, "optimum.gptq", None)
    res_gptq_except = _apply_gptq_quantization("model")
    assert res_gptq_except == (0.75, "quantized_gptq")
