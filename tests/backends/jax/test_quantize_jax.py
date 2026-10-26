"""Tests for JAX quantization logic, including uniform int8 and AWQ."""

from __future__ import annotations

from typing import Any

from tests.conftest import requires_jax

pytestmark = requires_jax


import jax.numpy as jnp
import pytest

import gemma_4_sql.backends.jax.quantize as qt
from gemma_4_sql.exceptions import DependencyMissingError, UnsupportedQuantizationMethodError


def test_quantize_int8_real_math() -> None:
    """Test uniform int8 quantization math and scale factor computation."""
    tensor = jnp.array([[-12.7, 0.0], [6.35, 12.7]], dtype=jnp.float32)
    q_tensor, scale = qt.quantize_int8(tensor)
    assert scale == pytest.approx(0.1)
    assert q_tensor[0, 0] == -127
    assert q_tensor[0, 1] == 0
    assert q_tensor[1, 0] in (63, 64)
    assert q_tensor[1, 1] == 127

    # Zero tensor edge case
    zero_tensor = jnp.zeros((2, 2), dtype=jnp.float32)
    q_zero, scale_zero = qt.quantize_int8(zero_tensor)
    assert scale_zero == 1.0
    assert jnp.all(q_zero == 0)


def test_compute_channel_activation_statistics() -> None:
    """Test channel activation magnitude statistics across token dimensions."""
    # Shape: (batch=2, seq_len=3, channels=4)
    activations = jnp.ones((2, 3, 4), dtype=jnp.float32) * 2.0
    stats = qt.compute_channel_activation_statistics(activations)
    assert stats.shape == (4,)
    assert jnp.allclose(stats, 2.0)

    # 1D already
    stats_1d = qt.compute_channel_activation_statistics(jnp.array([1.0, 5.0, 3.0]))
    assert stats_1d.shape == (3,)
    assert stats_1d[1] == pytest.approx(5.0)


def test_compute_salient_mask_and_bounds() -> None:
    """Test salient mask computation and ratio bounds validation."""
    magnitudes = jnp.array([0.1, 10.0, 0.2, 5.0, 0.3])
    # 20% of 5 is 1 channel
    mask = qt.compute_salient_mask(magnitudes, salient_ratio=0.2)
    assert bool(mask[1]) is True
    assert bool(mask[0]) is False

    # Test invalid ratio bounds
    with pytest.raises(ValueError, match="salient_ratio must be in"):
        qt.compute_salient_mask(magnitudes, salient_ratio=0.0)

    with pytest.raises(ValueError, match="salient_ratio must be in"):
        qt.compute_salient_mask(magnitudes, salient_ratio=1.5)


def test_quantize_awq_and_dequantize() -> None:
    """Test AWQ quantization, salient weight preservation, and dequantization."""
    # (in_features=4, out_features=4)
    tensor = jnp.array(
        [
            [100.0, 200.0, -100.0, -200.0],  # Channel 0: large salient weights
            [1.0, -1.0, 0.5, -0.5],  # Channel 1: normal weights
            [0.1, 0.2, -0.1, -0.2],  # Channel 2: small weights
            [0.05, -0.05, 0.05, -0.05],  # Channel 3: tiny weights
        ],
        dtype=jnp.float32,
    )
    # Channel 0 has largest activation
    channel_acts = jnp.array([10.0, 1.0, 0.1, 0.05])
    q_tensor, scale, salient_weights, salient_mask = qt.quantize_awq(
        tensor,
        channel_activations=channel_acts,
        salient_ratio=0.25,
    )

    assert bool(salient_mask[0]) is True
    assert jnp.all(salient_weights[0] == tensor[0])
    # Salient channel should be 0 in quantized tensor
    assert jnp.all(q_tensor[0] == 0)

    # Dequantize
    reconstructed = qt.dequantize_awq(q_tensor, scale, salient_weights)
    # Salient channel must have EXACT parity with zero error
    assert jnp.allclose(reconstructed[0], tensor[0])
    # Non-salient channels should be close within int8 quantization tolerance
    assert jnp.allclose(reconstructed[1:], tensor[1:], atol=0.05)


def test_quantize_awq_fallback_activations() -> None:
    """Test AWQ when channel_activations is None or multi-dimensional."""
    tensor = jnp.array([[10.0, 20.0], [1.0, 2.0]], dtype=jnp.float32)
    # Fallback to weight magnitude
    _q_tensor, _scale, _salient_weights, salient_mask = qt.quantize_awq(tensor, channel_activations=None, salient_ratio=0.5)
    assert salient_mask.shape == (2,)

    # Multi-dimensional activations passed
    acts = jnp.ones((2, 3, 2), dtype=jnp.float32)
    _q_tensor2, _scale2, _salient_weights2, salient_mask2 = qt.quantize_awq(tensor, channel_activations=acts, salient_ratio=0.5)
    assert salient_mask2.shape == (2,)

    # 1D activations fallback branch for different shape
    acts_1d_wrong = jnp.ones((4,), dtype=jnp.float32)
    _q_tensor3, _scale3, _salient_weights3, salient_mask3 = qt.quantize_awq(tensor, channel_activations=acts_1d_wrong, salient_ratio=0.5)
    assert salient_mask3.shape == (2,)

    # Zero non-salient weights scale handling
    zero_tensor = jnp.zeros((2, 2), dtype=jnp.float32)
    _q_z, s_z, _w_z, _m_z = qt.quantize_awq(zero_tensor, channel_activations=None, salient_ratio=0.5)
    assert s_z == 1.0


def test_quantize_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX quantize when dependencies are missing."""
    monkeypatch.setattr(qt, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing\."):
        qt.quantize_model("model", "int8")

    monkeypatch.setattr(qt, "jnp", None)
    with pytest.raises(DependencyMissingError, match="JAX is required for quantize_int8"):
        qt.quantize_int8(jnp.array([1.0, 2.0]))

    with pytest.raises(DependencyMissingError, match="JAX is required for compute_channel_activation_statistics"):
        qt.compute_channel_activation_statistics(jnp.array([1.0, 2.0]))

    with pytest.raises(DependencyMissingError, match="JAX is required for compute_salient_mask"):
        qt.compute_salient_mask(jnp.array([1.0, 2.0]))

    with pytest.raises(DependencyMissingError, match="JAX is required for quantize_awq"):
        qt.quantize_awq(jnp.array([1.0, 2.0]))

    with pytest.raises(DependencyMissingError, match="JAX is required for dequantize_awq"):
        qt.dequantize_awq(jnp.array([1.0]), jnp.array([1.0]), jnp.array([1.0]))


@pytest.fixture
def _mock_jax_quantize_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure `Gemma4Config` presets to return a tiny Gemma 4 model for fast real-model quantization tests."""
    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config

    tiny_cfg = Gemma4Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=1,
    )
    monkeypatch.setattr(qt.Gemma4Config, "gemma4_e2b", staticmethod(lambda: tiny_cfg))
    monkeypatch.setattr(qt.Gemma4Config, "from_preset", classmethod(lambda _cls, _name: tiny_cfg))


@pytest.mark.usefixtures("_mock_jax_quantize_env")
def test_quantize_jax_model_int8_and_awq() -> None:
    """Test quantize_model for both uniform int8 and AWQ methods."""
    res_int8 = qt.quantize_model("test_gemma", "int8")
    assert res_int8["backend"] == "jax"
    assert res_int8["status"] == "quantized_int8"
    assert res_int8["memory_reduction_factor"] == pytest.approx(0.5)

    # Test AWQ with calibration activations
    calib = [jnp.ones((1, 4, 64), dtype=jnp.float32), jnp.ones((1, 4, 64), dtype=jnp.float32)]
    res_awq = qt.quantize_model("test_gemma", "awq", calibration_samples=calib, salient_ratio=0.5)
    assert res_awq["backend"] == "jax"
    assert res_awq["status"] == "quantized_awq"
    assert res_awq["memory_reduction_factor"] == pytest.approx(0.7)

    # Test AWQ without calibration
    res_awq2 = qt.quantize_model("test_gemma", "awq", salient_ratio=0.5)
    assert res_awq2["backend"] == "jax"
    assert res_awq2["status"] == "quantized_awq"
    assert res_awq2["memory_reduction_factor"] == pytest.approx(0.7)

    # Test unsupported method
    res_unsupported = qt.quantize_model("test_gemma", "unsupported_method")
    assert res_unsupported["status"] == "unsupported_method_unsupported_method"
    assert res_unsupported["memory_reduction_factor"] == 0.0
    with pytest.raises(UnsupportedQuantizationMethodError):
        qt.quantize_model("test_gemma", "unsupported_method", strict_method=True)


@pytest.mark.usefixtures("_mock_jax_quantize_env")
def test_quantize_jax_model_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test quantize_model when an error occurs in model graph iteration."""

    def raise_runtime_err(*_args: Any, **_kwargs: Any) -> Any:
        """Provide raise runtime err test helper."""
        raise RuntimeError("Mock graph iteration failure")

    monkeypatch.setattr(qt, "_apply_quantization_to_model", raise_runtime_err)
    res = qt.quantize_model("test_gemma", "int8")
    assert "failed: Mock graph iteration failure" in res["status"]
    assert res["memory_reduction_factor"] == 0.0


def test_set_param_metadata_fallback() -> None:
    """Test _set_param_metadata fallback when set_metadata is not present."""

    class DummyParam:
        """Dummy parameter object without set_metadata method."""

    dummy: Any = DummyParam()
    qt._set_param_metadata(dummy, "custom_key", "custom_val")
    assert dummy.custom_key == "custom_val"


def test_set_param_metadata_native() -> None:
    """Test _set_param_metadata when set_metadata is present."""

    class DummyParamNative:
        """Test stub for `DummyParamNative`."""

        def set_metadata(self, k: str, v: str) -> None:
            """Provide set metadata test helper."""
            setattr(self, k, v)

    dummy: Any = DummyParamNative()
    qt._set_param_metadata(dummy, "custom_key", "custom_val")
    assert dummy.custom_key == "custom_val"


def test_quantize_real_tiny_jax_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify quantize_model runs AWQ quantization on a real tiny Gemma4ForCausalLM."""
    from flax import nnx

    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM

    cfg = Gemma4Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
    )
    monkeypatch.setattr(qt.Gemma4Config, "gemma4_e2b", staticmethod(lambda: cfg))
    monkeypatch.setattr(qt, "Gemma4ForCausalLM", Gemma4ForCausalLM)
    monkeypatch.setattr(qt, "nnx", nnx)

    res = qt.quantize_model("tiny-jax", "awq", salient_ratio=0.25)
    assert res["backend"] == "jax"
    assert res["status"] == "quantized_awq"
    assert res["memory_reduction_factor"] == pytest.approx(0.7)
