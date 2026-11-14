"""Unit and behavioral tests for `gemma_4_sql.backends.common_quantize`."""

from __future__ import annotations

from typing import Never

import pytest

from gemma_4_sql.backends.common_quantize import (
    ModelWithQuantizableConfigProtocol,
    QuantizableConfigHolderProtocol,
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
from gemma_4_sql.type_hints import DTypeProtocol, JSONValue, get_quantized_artifact


class _RecordingBnbConfigFactory:
    """Typed callable double recording `BitsAndBytesConfig` keyword arguments."""

    def __init__(self) -> None:
        """Initialize empty call log."""
        self.calls: list[dict[str, JSONValue | str | None]] = []

    def __call__(
        self,
        *,
        load_in_8bit: bool = False,
        llm_int8_threshold: float = 6.0,
        llm_int8_skip_modules: list[str] | None = None,
        load_in_4bit: bool = False,
        bnb_4bit_compute_dtype: DTypeProtocol | type[int | float | bool] | str | None = None,
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_quant_type: str = "nf4",
    ) -> dict[str, JSONValue]:
        """Record quantization configuration options and return a JSON config dictionary.

        Args:
            load_in_8bit: Whether 8-bit quantization is enabled.
            llm_int8_threshold: Outlier threshold for 8-bit features.
            llm_int8_skip_modules: Optional list of module names to skip.
            load_in_4bit: Whether 4-bit quantization is enabled.
            bnb_4bit_compute_dtype: Compute dtype for 4-bit dequantization.
            bnb_4bit_use_double_quant: Whether double quantization is enabled.
            bnb_4bit_quant_type: 4-bit quantization format (`'nf4'` or `'fp4'`).

        Returns:
            Dictionary of normalized BitsAndBytes settings.

        """
        dtype_str = str(bnb_4bit_compute_dtype) if bnb_4bit_compute_dtype is not None else None
        payload: dict[str, JSONValue] = {
            "load_in_8bit": load_in_8bit,
            "llm_int8_threshold": llm_int8_threshold,
            "llm_int8_skip_modules": llm_int8_skip_modules,
            "load_in_4bit": load_in_4bit,
            "bnb_4bit_compute_dtype": dtype_str,
            "bnb_4bit_use_double_quant": bnb_4bit_use_double_quant,
            "bnb_4bit_quant_type": bnb_4bit_quant_type,
        }
        self.calls.append(payload)
        return payload


class _FakeQuantConfig:
    """Typed `QuantizableConfigHolderProtocol` implementation."""

    def __init__(self) -> None:
        """Initialize with `quantization_config = None`."""
        self.quantization_config: JSONValue | None = None


class _FakeModelWithConfig:
    """Typed `ModelWithQuantizableConfigProtocol` implementation."""

    def __init__(self, config: _FakeQuantConfig | None) -> None:
        """Initialize with optional `_FakeQuantConfig`.

        Args:
            config: Configuration holder or `None`.

        """
        self._config = config

    @property
    def config(self) -> _FakeQuantConfig | None:
        """Provide the attached configuration holder.

        Returns:
            `_FakeQuantConfig` or `None`.

        """
        return self._config


def test_apply_bits_and_bytes_quantization_missing_deps() -> None:
    """Verify `apply_bits_and_bytes_quantization` raises `DependencyMissingError` when config class is `None`."""
    with pytest.raises(DependencyMissingError):
        apply_bits_and_bytes_quantization("int8", None)


def test_apply_bits_and_bytes_quantization_int8() -> None:
    """Verify `apply_bits_and_bytes_quantization` builds INT8 configs with and without `llm_int8_skip_modules`."""
    factory = _RecordingBnbConfigFactory()
    reduction, status = apply_bits_and_bytes_quantization("int8", factory, llm_int8_skip_modules=["skipme"])
    assert status == "quantized_int8"
    assert reduction == 0.5
    assert factory.calls[-1]["load_in_8bit"] is True
    assert factory.calls[-1]["llm_int8_skip_modules"] == ["skipme"]

    reduction, status = apply_bits_and_bytes_quantization("int8", factory)
    assert status == "quantized_int8"
    assert reduction == 0.5
    assert factory.calls[-1]["llm_int8_skip_modules"] is None


def test_apply_bits_and_bytes_quantization_int4() -> None:
    """Verify `apply_bits_and_bytes_quantization` builds INT4 NF4 configs."""
    factory = _RecordingBnbConfigFactory()
    reduction, status = apply_bits_and_bytes_quantization("int4", factory, "float16")
    assert status == "quantized_int4"
    assert reduction == 0.75
    assert factory.calls[-1]["load_in_4bit"] is True
    assert factory.calls[-1]["bnb_4bit_compute_dtype"] == "float16"
    assert factory.calls[-1]["bnb_4bit_use_double_quant"] is True
    assert factory.calls[-1]["bnb_4bit_quant_type"] == "nf4"


def test_apply_bits_and_bytes_quantization_unsupported() -> None:
    """Verify `apply_bits_and_bytes_quantization` raises `UnsupportedQuantizationMethodError` on unknown methods."""
    factory = _RecordingBnbConfigFactory()
    with pytest.raises(
        UnsupportedQuantizationMethodError, match="Unsupported quantization method for BitsAndBytes: unknown"
    ):
        apply_bits_and_bytes_quantization("unknown", factory)


def test_apply_bits_and_bytes_quantization_with_model() -> None:
    """Verify `apply_bits_and_bytes_quantization` attaches config and registers `QuantizedModelArtifact`."""
    factory = _RecordingBnbConfigFactory()
    cfg = _FakeQuantConfig()
    assert isinstance(cfg, QuantizableConfigHolderProtocol)
    model_with_cfg = _FakeModelWithConfig(cfg)
    assert isinstance(model_with_cfg, ModelWithQuantizableConfigProtocol)

    apply_bits_and_bytes_quantization("int8", factory, model=model_with_cfg)
    artifact = get_quantized_artifact(model_with_cfg)
    assert artifact is not None
    assert artifact.is_quantized is True
    assert artifact.method == "int8"
    assert cfg.quantization_config is not None

    model_no_cfg = _FakeModelWithConfig(None)
    apply_bits_and_bytes_quantization("int8", factory, model=model_no_cfg)
    assert get_quantized_artifact(model_no_cfg) is not None


def test_quantize_model_wrapper() -> None:
    """Verify `quantize_model_wrapper` handles success, unsupported status, operational failure, and raise_on_error."""
    res = quantize_model_wrapper("b", "m", "int8", lambda: (0.5, "ok"))
    assert res["status"] == "ok"

    res = quantize_model_wrapper("b", "m", "int8", lambda: (0.0, "unsupported_xyz"))
    assert res["status"] == "unsupported_xyz"

    def fail_fn() -> Never:
        """Raise ValueError to test operational failure capture.

        Raises:
            ValueError: Always raised for test verification.

        """
        raise ValueError("oops")

    res = quantize_model_wrapper("b", "m", "int8", fail_fn)
    assert res["status"] == "failed: oops"

    with pytest.raises(QuantizationError, match="oops"):
        quantize_model_wrapper("b", "m", "int8", fail_fn, raise_on_error=True)

    def unsupported_fn() -> Never:
        """Raise `UnsupportedQuantizationMethodError`.

        Raises:
            UnsupportedQuantizationMethodError: Always raised for test verification.

        """
        raise UnsupportedQuantizationMethodError("bad method")

    with pytest.raises(UnsupportedQuantizationMethodError, match="bad method"):
        quantize_model_wrapper("b", "m", "int8", unsupported_fn)


def test_compute_awq_and_gptq_calibration() -> None:
    """Verify `compute_awq_channel_scales` and `compute_gptq_compensated_weights` across valid and edge-case inputs."""
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

    from gemma_4_sql.backends.common_quantize import _invert_spd_matrix

    assert _invert_spd_matrix([]) == []
    spd = [[4.0, 2.0], [2.0, 3.0]]
    spd_inv = _invert_spd_matrix(spd)
    # Verify spd @ spd_inv == I_2
    assert spd[0][0] * spd_inv[0][0] + spd[0][1] * spd_inv[1][0] == pytest.approx(1.0, abs=1e-6)
    assert spd[0][0] * spd_inv[0][1] + spd[0][1] * spd_inv[1][1] == pytest.approx(0.0, abs=1e-6)
    assert spd[1][0] * spd_inv[0][1] + spd[1][1] * spd_inv[1][1] == pytest.approx(1.0, abs=1e-6)

    # Non-SPD matrix requiring row swap and negative pivot handling in _invert_with_partial_pivoting
    indefinite = [[0.0, 2.0], [-4.0, 1.0]]
    indef_inv = _invert_spd_matrix(indefinite)
    assert indefinite[0][0] * indef_inv[0][0] + indefinite[0][1] * indef_inv[1][0] == pytest.approx(1.0, abs=1e-6)
    assert indefinite[1][0] * indef_inv[0][1] + indefinite[1][1] * indef_inv[1][1] == pytest.approx(1.0, abs=1e-6)

    # Near-zero negative diagonal clamping branch
    singular_neg = [[-1e-12]]
    assert len(_invert_spd_matrix(singular_neg)) == 1

    # Native 8x8 SPD matrix inversion (with torch loaded, indefinite fallback, and torch absent)
    import sys

    import torch

    assert torch is not None

    spd8 = [[10.0 if i == j else 0.5 for j in range(8)] for i in range(8)]
    inv8 = _invert_spd_matrix(spd8)
    assert len(inv8) == 8
    assert sum(spd8[0][k] * inv8[k][0] for k in range(8)) == pytest.approx(1.0, abs=1e-6)

    indef8 = [[-10.0 if i == 0 and j == 0 else (4.0 if i == j else 0.0) for j in range(8)] for i in range(8)]
    inv_indef8 = _invert_spd_matrix(indef8)
    assert inv_indef8[0][0] == pytest.approx(-0.1, abs=1e-6)

    from unittest.mock import patch

    with patch.dict(sys.modules, {"torch": None}):
        inv8_py = _invert_spd_matrix(spd8)
        assert sum(spd8[0][k] * inv8_py[k][0] for k in range(8)) == pytest.approx(1.0, abs=1e-6)
