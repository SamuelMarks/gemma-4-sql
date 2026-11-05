"""Unit and behavioral tests for common quantize."""

from typing import Never
from unittest.mock import MagicMock

import pytest

from gemma_4_sql.backends.common_quantize import (
    _is_config_model,
    apply_bits_and_bytes_quantization,
    compute_awq_channel_scales,
    compute_gptq_compensated_weights,
    quantize_model_wrapper,
)
from gemma_4_sql.exceptions import (
    DependencyMissingError,
    QuantizationError,
    UnsupportedQuantizationMethodError,
)
from gemma_4_sql.type_hints import get_quantized_artifact


def test_apply_bits_and_bytes_quantization_missing_deps() -> None:
    """Verify apply bits and bytes quantization missing deps behavior."""
    with pytest.raises(DependencyMissingError):
        apply_bits_and_bytes_quantization("int8", None)


def test_apply_bits_and_bytes_quantization_int8() -> None:
    """Verify apply bits and bytes quantization int8 behavior."""
    mock_cls = MagicMock()
    mock_cls.return_value = "config_obj"
    reduction, status = apply_bits_and_bytes_quantization("int8", mock_cls, llm_int8_skip_modules=["skipme"])
    assert status == "quantized_int8"
    assert reduction == 0.5
    mock_cls.assert_called_with(load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_skip_modules=["skipme"])

    reduction, status = apply_bits_and_bytes_quantization("int8", mock_cls)
    assert status == "quantized_int8"
    assert reduction == 0.5


def test_apply_bits_and_bytes_quantization_int4() -> None:
    """Verify apply bits and bytes quantization int4 behavior."""
    mock_cls = MagicMock()
    mock_cls.return_value = "config_obj"
    reduction, status = apply_bits_and_bytes_quantization("int4", mock_cls, "float16")
    assert status == "quantized_int4"
    assert reduction == 0.75
    mock_cls.assert_called_with(load_in_4bit=True, bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")


def test_apply_bits_and_bytes_quantization_unsupported() -> None:
    """Verify apply bits and bytes quantization unsupported behavior."""
    mock_cls = MagicMock()
    with pytest.raises(UnsupportedQuantizationMethodError, match="Unsupported quantization method for BitsAndBytes: unknown"):
        apply_bits_and_bytes_quantization("unknown", mock_cls)


def test_apply_bits_and_bytes_quantization_with_model() -> None:
    """Verify apply bits and bytes quantization with model behavior."""
    mock_cls = MagicMock()
    mock_model = MagicMock()
    mock_model.config = MagicMock()
    apply_bits_and_bytes_quantization("int8", mock_cls, model=mock_model)
    artifact = get_quantized_artifact(mock_model)
    assert artifact is not None
    assert artifact.is_quantized is True
    assert artifact.method == "int8"
    assert mock_model.config.quantization_config is not None

    mock_model_no_cfg = MagicMock()
    mock_model_no_cfg.config = None
    apply_bits_and_bytes_quantization("int8", mock_cls, model=mock_model_no_cfg)
    assert get_quantized_artifact(mock_model_no_cfg) is not None
    assert _is_config_model(mock_model) is True
    assert _is_config_model(None) is False


def test_quantize_model_wrapper() -> None:
    """Verify quantize model wrapper behavior."""
    res = quantize_model_wrapper("b", "m", "int8", lambda: (0.5, "ok"))
    assert res["status"] == "ok"

    res = quantize_model_wrapper("b", "m", "int8", lambda: (0.0, "unsupported_xyz"))
    assert res["status"] == "unsupported_xyz"

    def fail_fn() -> Never:
        """Provide fail fn test helper."""
        raise ValueError("oops")

    res = quantize_model_wrapper("b", "m", "int8", fail_fn)
    assert res["status"] == "failed: oops"

    with pytest.raises(QuantizationError, match="oops"):
        quantize_model_wrapper("b", "m", "int8", fail_fn, raise_on_error=True)

    def unsupported_fn() -> Never:
        """Raise UnsupportedQuantizationMethodError."""
        raise UnsupportedQuantizationMethodError("bad method")

    with pytest.raises(UnsupportedQuantizationMethodError, match="bad method"):
        quantize_model_wrapper("b", "m", "int8", unsupported_fn)


def test_compute_awq_and_gptq_calibration() -> None:
    """Verify compute_awq_channel_scales and compute_gptq_compensated_weights across valid and edge-case inputs."""
    w = [[0.5, -1.2, 0.3], [1.1, 0.4, -0.8]]
    acts = [[1.0, 2.0, 0.5], [0.5, 4.0, 1.0]]

    scales_with_act = compute_awq_channel_scales(w, acts)
    assert len(scales_with_act) == 3
    assert all(s > 0.0 for s in scales_with_act)

    scales_no_act = compute_awq_channel_scales(w, None)
    assert len(scales_no_act) == 3

    with pytest.raises(ValueError, match="alpha_range"):
        compute_awq_channel_scales(w, acts, alpha_range=())

    with pytest.raises(ValueError, match="weight_2d"):
        compute_awq_channel_scales([], acts)

    gptq_with_act = compute_gptq_compensated_weights(w, acts, bits=4)
    assert len(gptq_with_act) == 2
    assert len(gptq_with_act[0]) == 3

    gptq_no_act = compute_gptq_compensated_weights(w, None, bits=4, damp_percent=-1.0)
    assert len(gptq_no_act) == 2
    assert len(gptq_no_act[0]) == 3
    assert compute_gptq_compensated_weights([], None) == []
