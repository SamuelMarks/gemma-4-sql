"""Tests for common quantize utilities."""

from unittest.mock import MagicMock

import pytest

from gemma_4_sql.backends.common_quantize import (
    apply_bits_and_bytes_quantization,
    quantize_model_wrapper,
)
from gemma_4_sql.exceptions import DependencyMissingError


def test_apply_bits_and_bytes_quantization_missing():
    """Docstring for test_apply_bits_and_bytes_quantization_missing."""
    res = apply_bits_and_bytes_quantization("int8", None)
    assert res == (0.0, "mocked_missing_bitsandbytes")

    with pytest.raises(DependencyMissingError):
        apply_bits_and_bytes_quantization("int8", None, raise_if_missing=True)


def test_apply_bits_and_bytes_quantization_int8():
    """Docstring for test_apply_bits_and_bytes_quantization_int8."""
    mock_cls = MagicMock()
    mock_cls.return_value = "mock_config"
    mock_model = MagicMock()
    mock_model.config.quantization_config = None

    res = apply_bits_and_bytes_quantization("int8", mock_cls, model=mock_model, llm_int8_skip_modules=["test"])
    assert res == (0.5, "quantized_int8")
    mock_cls.assert_called_once_with(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=["test"],
    )
    assert mock_model.config.quantization_config == "mock_config"
    assert mock_model._is_quantized is True
    assert mock_model._quant_method == "int8"


def test_apply_bits_and_bytes_quantization_int4():
    """Docstring for test_apply_bits_and_bytes_quantization_int4."""
    mock_cls = MagicMock()
    res = apply_bits_and_bytes_quantization("int4", mock_cls, "float16")
    assert res == (0.75, "quantized_int4")
    mock_cls.assert_called_once_with(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def test_apply_bits_and_bytes_quantization_unsupported():
    """Docstring for test_apply_bits_and_bytes_quantization_unsupported."""
    mock_cls = MagicMock()
    res = apply_bits_and_bytes_quantization("int16", mock_cls)
    assert res == (0.0, "unsupported_method_int16")


def test_quantize_model_wrapper_missing():
    """Docstring for test_quantize_model_wrapper_missing."""
    res = quantize_model_wrapper("test_backend", "model", "method", True, "missing", lambda: (0.0, "status"))
    assert res["status"] == "missing"
    assert res["memory_reduction_factor"] == 0.0


def test_quantize_model_wrapper_success():
    """Docstring for test_quantize_model_wrapper_success."""
    res = quantize_model_wrapper("test_backend", "model", "method", False, "missing", lambda: (0.5, "quantized_method"))
    assert res["status"] == "quantized_method"
    assert res["memory_reduction_factor"] == 0.5


def test_quantize_model_wrapper_failure():
    """Docstring for test_quantize_model_wrapper_failure."""

    def mock_fail():
        """Docstring for mock_fail."""
        raise RuntimeError("test error")

    res = quantize_model_wrapper("test_backend", "model", "method", False, "missing", mock_fail)
    assert res["status"] == "failed: test error"
    assert res["memory_reduction_factor"] == 0.0


def test_apply_bits_and_bytes_quantization_int8_no_skip():
    """Docstring for test_apply_bits_and_bytes_quantization_int8_no_skip."""
    mock_cls = MagicMock()
    mock_cls.return_value = "mock_config"
    res = apply_bits_and_bytes_quantization("int8", mock_cls)
    assert res == (0.5, "quantized_int8")
    mock_cls.assert_called_once_with(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )


def test_apply_bits_and_bytes_quantization_model_no_config():
    """Docstring for test_apply_bits_and_bytes_quantization_model_no_config."""
    mock_cls = MagicMock()

    class DummyModel:
        """Docstring for DummyModel."""

    dummy = DummyModel()
    res = apply_bits_and_bytes_quantization("int4", mock_cls, model=dummy)
    assert res == (0.75, "quantized_int4")
    assert dummy._is_quantized is True
