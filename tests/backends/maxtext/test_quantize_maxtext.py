"""Tests for MaxText AQT quantization logic."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import gemma_4_sql.backends.maxtext.quantize as maxtext_quantize
from gemma_4_sql.backends.maxtext.quantize import (
    apply_aqt_quantization,
    quantize_model,
    quantize_tensor_aqt,
)
from gemma_4_sql.exceptions import DependencyMissingError


class MockJnp:
    """Mock JAX numpy."""

    int32 = 1

    @staticmethod
    def zeros(_shape: object, **_kwargs: object) -> object:
        """Return dummy zero list."""
        return [0]


class MockJaxRandom:
    """Mock JAX random module."""

    @staticmethod
    def mock_prngkey(seed: object) -> object:
        """Return dummy PRNGKey."""
        return seed

    PRNGKey = mock_prngkey


class MockJax:
    """Mock JAX module."""

    random = MockJaxRandom()


class MockGemma4Model:
    """Mock Gemma4Model."""

    def __init__(self, name: object) -> None:
        """Initialize mock model."""

    def init(self, _rng: object, _inputs: object) -> object:
        """Return mock parameter string."""
        return "params"


def test_quantize_tensor_aqt_int8_and_int4() -> None:
    """Test quantize_tensor_aqt numerical properties for int8 and int4."""
    w = jnp.array([[2.0, -1.0, 0.5], [-3.0, 1.5, 0.0]], dtype=jnp.float32)

    # Test int8
    q8, s8 = quantize_tensor_aqt(w, bits=8)
    assert q8.dtype == jnp.int8
    assert s8.shape == (2, 1)
    # Clipping bound is 127
    recon8 = q8.astype(jnp.float32) * s8
    assert jnp.max(jnp.abs(w - recon8)) < 0.05

    # Test int4
    q4, s4 = quantize_tensor_aqt(w, bits=4)
    # Clipping bound is 7
    assert jnp.max(q4) <= 7
    assert jnp.min(q4) >= -8
    assert s4.shape == (2, 1)


def test_quantize_tensor_aqt_invalid_bits_and_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test quantize_tensor_aqt raises errors on invalid bits or missing dependencies."""
    with pytest.raises(ValueError, match="Quantization bits must be positive"):
        quantize_tensor_aqt(jnp.ones((2, 2)), bits=0)

    monkeypatch.setattr(maxtext_quantize, "jax", None)
    with pytest.raises(DependencyMissingError, match="JAX dependencies are missing"):
        quantize_tensor_aqt(jnp.ones((2, 2)), bits=8)


def test_apply_aqt_quantization_nested_params() -> None:
    """Test apply_aqt_quantization across nested model parameters."""
    params = {
        "decoder": {
            "layers_0": {
                "q_proj": {"kernel": jnp.ones((8, 16))},
                "out_proj": {"kernel": jnp.ones((16, 8))},
                "bias": jnp.zeros((8,)),
            }
        }
    }
    # Test int8 with custom targets
    q_params, meta, count = apply_aqt_quantization(params, method="int8", quant_targets=["q_proj"])
    assert count == 1
    assert meta["bits"] == 8
    assert meta["clipping_bound"] == 127.0
    assert meta["memory_reduction_factor"] == 0.5

    q_proj = q_params["decoder"]["layers_0"]["q_proj"]
    assert "kernel_scale" in q_proj
    assert q_proj["kernel"].dtype == jnp.int8
    assert q_proj["aqt_config"]["bits"] == 8

    # out_proj should remain untouched
    out_proj = q_params["decoder"]["layers_0"]["out_proj"]
    assert "kernel_scale" not in out_proj

    # Test int4
    _q_params4, meta4, count4 = apply_aqt_quantization(params, method="int4", quant_targets=["q_proj", "out_proj"])
    assert count4 == 2
    assert meta4["bits"] == 4
    assert meta4["clipping_bound"] == 7.0
    assert meta4["memory_reduction_factor"] == 0.75

    # Test unknown method
    _, meta_other, _ = apply_aqt_quantization(params, method="other")
    assert meta_other["memory_reduction_factor"] == 0.7

    # Test non-dict params
    p_non_dict, _meta_nd, c_nd = apply_aqt_quantization("not_dict", method="int8")
    assert c_nd == 0
    assert p_non_dict == "not_dict"


def test_apply_aqt_quantization_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test apply_aqt_quantization raises DependencyMissingError when JAX is absent."""
    monkeypatch.setattr(maxtext_quantize, "jax", None)
    with pytest.raises(DependencyMissingError, match="JAX dependencies are missing"):
        apply_aqt_quantization({}, method="int8")


def test_quantize_maxtext_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MaxText quantize raises DependencyMissingError when JAX is missing."""
    monkeypatch.setattr(maxtext_quantize, "jnp", None)
    with pytest.raises(DependencyMissingError):
        quantize_model("model", "int8")


def test_quantize_maxtext_mocked_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MaxText quantize with mock model."""
    monkeypatch.setattr(maxtext_quantize, "jax", MockJax())
    monkeypatch.setattr(maxtext_quantize, "jnp", MockJnp())
    monkeypatch.setattr(maxtext_quantize, "Gemma4Model", MockGemma4Model)

    res8 = quantize_model("model", "int8")
    assert res8["backend"] == "maxtext"
    assert res8["status"] == "quantized_int8"
    assert res8["memory_reduction_factor"] == 0.5

    res4 = quantize_model("model", "int4")
    assert res4["status"] == "quantized_int4"
    assert res4["memory_reduction_factor"] == 0.75

    res_other = quantize_model("model", "awq")
    assert res_other["method"] == "awq"
    assert res_other["status"] == "quantized_awq"


def test_quantize_maxtext_with_real_params() -> None:
    """Test quantize_model with genuine parameter dict passed in kwargs."""
    params = {
        "decoder": {
            "layers_0": {
                "q_proj": {"kernel": jnp.ones((4, 4))},
                "v_proj": {"kernel": jnp.ones((4, 4))},
            }
        }
    }
    res = quantize_model("gemma-4", method="int8", params=params, quant_targets=["q_proj", "v_proj"])
    assert res["status"] == "quantized_int8"
    assert res["memory_reduction_factor"] == 0.5
    assert "metadata" in res
    assert res["metadata"]["quantized_modules_count"] == 2


def test_quantize_maxtext_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test quantize_model captures errors cleanly."""
    monkeypatch.setattr(maxtext_quantize, "jax", MockJax())
    monkeypatch.setattr(maxtext_quantize, "jnp", MockJnp())
    monkeypatch.setattr(maxtext_quantize, "Gemma4Model", MockGemma4Model)

    def raise_err(*_args: object, **_kwargs: object) -> object:
        msg = "init failed"
        raise ValueError(msg)

    monkeypatch.setattr(MockJnp, "zeros", raise_err)
    res = quantize_model("model", "int8")
    assert "failed: init failed" in str(res["status"])


def test_quantize_with_aqt_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test AQT metadata when native aqt library is present."""
    mock_aqt = type("MockAQT", (), {})
    monkeypatch.setattr(maxtext_quantize, "aqt", mock_aqt)
    _, meta, _ = apply_aqt_quantization({}, method="int8")
    assert meta["aqt_native"] is True


def test_quantize_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test module reloading when JAX is missing or AQT is present."""
    import importlib
    import sys

    m_quantize = sys.modules["gemma_4_sql.backends.maxtext.quantize"]

    # Test with mock aqt in sys.modules
    mock_v2 = type("MockV2", (), {})
    mock_jax = type("MockJ", (), {"v2": mock_v2})
    mock_aqt = type("MockA", (), {"jax": mock_jax})
    monkeypatch.setitem(sys.modules, "aqt", mock_aqt)
    monkeypatch.setitem(sys.modules, "aqt.jax", mock_jax)
    monkeypatch.setitem(sys.modules, "aqt.jax.v2", mock_v2)
    importlib.reload(m_quantize)
    assert m_quantize.aqt is not None

    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(m_quantize)
    monkeypatch.undo()
    importlib.reload(m_quantize)
