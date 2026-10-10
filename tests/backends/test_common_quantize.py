"""Module docstring."""

from unittest.mock import MagicMock

import pytest

from gemma_4_sql.backends.common_quantize import apply_bits_and_bytes_quantization, quantize_model_wrapper
from gemma_4_sql.exceptions import DependencyMissingError


def test_apply_bits_and_bytes_quantization_missing_deps():
    """Docstring for test_apply_bits_and_bytes_quantization_missing_deps."""
    with pytest.raises(DependencyMissingError):
        apply_bits_and_bytes_quantization("int8", None, raise_if_missing=True)

    reduction, status = apply_bits_and_bytes_quantization("int8", None, raise_if_missing=False)
    assert status == "mocked_missing_bitsandbytes"


def test_apply_bits_and_bytes_quantization_int8():
    """Docstring for test_apply_bits_and_bytes_quantization_int8."""
    mock_cls = MagicMock()
    mock_cls.return_value = "config_obj"
    reduction, status = apply_bits_and_bytes_quantization("int8", mock_cls, llm_int8_skip_modules=["skipme"])
    assert status == "quantized_int8"
    assert reduction == 0.5
    mock_cls.assert_called_with(load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_skip_modules=["skipme"])

    reduction, status = apply_bits_and_bytes_quantization("int8", mock_cls)
    assert status == "quantized_int8"
    assert reduction == 0.5


def test_apply_bits_and_bytes_quantization_int4():
    """Docstring for test_apply_bits_and_bytes_quantization_int4."""
    mock_cls = MagicMock()
    mock_cls.return_value = "config_obj"
    reduction, status = apply_bits_and_bytes_quantization("int4", mock_cls, "float16")
    assert status == "quantized_int4"
    assert reduction == 0.75
    mock_cls.assert_called_with(load_in_4bit=True, bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")


def test_apply_bits_and_bytes_quantization_unsupported():
    """Docstring for test_apply_bits_and_bytes_quantization_unsupported."""
    mock_cls = MagicMock()
    reduction, status = apply_bits_and_bytes_quantization("unknown", mock_cls)
    assert status == "unsupported_method_unknown"


def test_apply_bits_and_bytes_quantization_with_model():
    """Docstring for test_apply_bits_and_bytes_quantization_with_model."""
    mock_cls = MagicMock()
    mock_model = MagicMock()
    mock_model.config = MagicMock()
    apply_bits_and_bytes_quantization("int8", mock_cls, model=mock_model)
    assert mock_model._is_quantized is True
    assert mock_model._quant_method == "int8"
    assert hasattr(mock_model.config, "quantization_config")


def test_quantize_model_wrapper():
    """Docstring for test_quantize_model_wrapper."""
    res = quantize_model_wrapper("b", "m", "int8", True, "missing", lambda: (0, ""))
    assert res["status"] == "missing"

    res = quantize_model_wrapper("b", "m", "int8", False, "", lambda: (0.5, "ok"))
    assert res["status"] == "ok"

    res = quantize_model_wrapper("b", "m", "int8", False, "", lambda: (0.0, "unsupported_xyz"))
    assert res["status"] == "unsupported_xyz"

    def fail_fn():
        """Docstring for fail_fn."""
        raise ValueError("oops")

    res = quantize_model_wrapper("b", "m", "int8", False, "", fail_fn)
    assert res["status"] == "failed: oops"
