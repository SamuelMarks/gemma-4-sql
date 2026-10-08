"""Module docstring."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import gemma_4_sql.backends.maxtext.quantize as quantize_mod
from gemma_4_sql.exceptions import DependencyMissingError


@pytest.fixture(autouse=True)
def reload_module_after_test():
    """Docstring for reload_module_after_test."""
    # Store original state to restore later if needed, but pytest runs in a process
    # We just ensure the module is reloaded to its normal state before leaving.
    yield
    importlib.reload(quantize_mod)


def test_module_imports_success():
    """Docstring for test_module_imports_success."""
    with patch.dict(
        sys.modules,
        {
            "jax": MagicMock(),
            "jax.numpy": MagicMock(),
            "aqt": MagicMock(),
            "aqt.jax": MagicMock(),
            "aqt.jax.v2": MagicMock(),
            "maxtext": MagicMock(),
            "maxtext.models": MagicMock(),
            "maxtext.models.gemma4": MagicMock(Gemma4Model=MagicMock()),
        },
    ):
        importlib.reload(quantize_mod)
        assert quantize_mod.jax is not None
        assert quantize_mod.jnp is not None
        assert quantize_mod.aqt is not None
        assert quantize_mod.Gemma4Model is not None


def test_module_imports_failure():
    """Docstring for test_module_imports_failure."""
    with patch.dict(
        sys.modules,
        {
            "jax": None,
            "aqt": None,
            "maxtext": None,
        },
    ):
        importlib.reload(quantize_mod)
        assert quantize_mod.jax is None
        assert quantize_mod.jnp is None
        assert quantize_mod.aqt is None
        assert quantize_mod.Gemma4Model is None


def test_quantize_tensor_aqt_missing_jax():
    """Docstring for test_quantize_tensor_aqt_missing_jax."""
    quantize_mod.jax = None
    quantize_mod.jnp = None
    with pytest.raises(DependencyMissingError, match="JAX dependencies are missing."):
        quantize_mod.quantize_tensor_aqt(None)


def test_quantize_tensor_aqt_negative_bits():
    """Docstring for test_quantize_tensor_aqt_negative_bits."""
    quantize_mod.jax = MagicMock()
    quantize_mod.jnp = MagicMock()
    with pytest.raises(ValueError, match="Quantization bits must be positive, got 0"):
        quantize_mod.quantize_tensor_aqt(MagicMock(), bits=0)
    with pytest.raises(ValueError, match="Quantization bits must be positive, got -1"):
        quantize_mod.quantize_tensor_aqt(MagicMock(), bits=-1)


def test_quantize_tensor_aqt_int8():
    """Docstring for test_quantize_tensor_aqt_int8."""
    quantize_mod.jax = MagicMock()
    quantize_mod.jnp = np

    tensor = np.array([[100.0, 50.0], [-10.0, -100.0]])
    quantized, scale = quantize_mod.quantize_tensor_aqt(tensor, bits=8)

    # clipping_bound = 127
    # max_val = [[100.0], [100.0]]
    # scale = max_val / 127 = [[0.78740157], [0.78740157]]
    # tensor / scale = [[127.0, 63.5], [-12.7, -127.0]]
    assert quantized.dtype == np.int8
    assert quantized.shape == (2, 2)
    assert scale.shape == (2, 1)


def test_quantize_tensor_aqt_int16():
    """Docstring for test_quantize_tensor_aqt_int16."""
    quantize_mod.jax = MagicMock()
    quantize_mod.jnp = np

    tensor = np.array([[1000.0, 500.0]])
    quantized, _scale = quantize_mod.quantize_tensor_aqt(tensor, bits=16)

    assert quantized.dtype == np.int16


def test_apply_aqt_quantization_missing_jax():
    """Docstring for test_apply_aqt_quantization_missing_jax."""
    quantize_mod.jax = None
    quantize_mod.jnp = None
    with pytest.raises(DependencyMissingError, match="JAX dependencies are missing."):
        quantize_mod.apply_aqt_quantization({})


def test_apply_aqt_quantization_methods():
    """Docstring for test_apply_aqt_quantization_methods."""
    quantize_mod.jax = MagicMock()
    quantize_mod.jnp = np

    params = {}

    # int8
    _, meta, _ = quantize_mod.apply_aqt_quantization(params, method="int8")
    assert meta["bits"] == 8
    assert meta["memory_reduction_factor"] == 0.5

    # int4
    _, meta, _ = quantize_mod.apply_aqt_quantization(params, method="int4")
    assert meta["bits"] == 4
    assert meta["memory_reduction_factor"] == 0.75

    # other
    _, meta, _ = quantize_mod.apply_aqt_quantization(params, method="unknown")
    assert meta["bits"] == 8
    assert meta["memory_reduction_factor"] == 0.7


def test_apply_aqt_quantization_targets_and_traverse():
    """Docstring for test_apply_aqt_quantization_targets_and_traverse."""
    quantize_mod.jax = MagicMock()
    quantize_mod.jnp = np
    quantize_mod.aqt = MagicMock()

    tensor = np.array([[1.0]])

    params = {"non_dict": "value", "nested": {"q_proj": {"kernel": tensor, "other_param": 42}, "unrelated": {"kernel": tensor}, "k_proj": {"kernel": tensor}}, "deep": {"layer": {"v_proj": {"kernel": tensor}}}}

    # custom targets
    q_params, _meta, count = quantize_mod.apply_aqt_quantization(params, quant_targets=["q_proj"])

    assert count == 1
    assert "kernel_scale" in q_params["nested"]["q_proj"]
    assert q_params["nested"]["q_proj"]["other_param"] == 42
    assert "kernel_scale" not in q_params["nested"]["k_proj"]
    assert "kernel_scale" not in q_params["deep"]["layer"]["v_proj"]
    assert q_params["non_dict"] == "value"

    # default targets
    _q_params2, meta2, count2 = quantize_mod.apply_aqt_quantization(params)
    assert count2 == 3  # q_proj, k_proj, v_proj
    assert meta2["aqt_native"] is True


def test_quantize_model_missing_maxtext_deps():
    """Docstring for test_quantize_model_missing_maxtext_deps."""
    quantize_mod.jax = None
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing."):
        quantize_mod.quantize_model("model")

    quantize_mod.jax = MagicMock()
    quantize_mod.jnp = None
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing."):
        quantize_mod.quantize_model("model")

    quantize_mod.jax = MagicMock()
    quantize_mod.jnp = np
    quantize_mod.Gemma4Model = None
    # No params provided
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing."):
        quantize_mod.quantize_model("model")

    # Gemma4Model missing but inside try block
    # We can hit this if we mock Gemma4Model to None *inside* the function if we bypass the outer check
    # But outer check is `jax is None or jnp is None or (Gemma4Model is None and "params" not in kwargs)`
    # We can pass params to bypass outer check, then remove params inside? No, it's kwargs.
    # To hit the inner `DependencyMissingError("MaxText dependency missing.")`, we provide params=None!
    with pytest.raises(DependencyMissingError, match="MaxText dependency missing."):
        quantize_mod.quantize_model("model", params=None)


def test_quantize_model_with_gemma4_init():
    """Docstring for test_quantize_model_with_gemma4_init."""
    quantize_mod.jax = MagicMock()
    quantize_mod.jnp = np
    quantize_mod.Gemma4Model = MagicMock()

    model_mock = MagicMock()
    quantize_mod.Gemma4Model.return_value = model_mock

    tensor = np.array([[1.0]])
    model_mock.init.return_value = {"q_proj": {"kernel": tensor}}

    res = quantize_mod.quantize_model("my_model")

    assert res["backend"] == "maxtext"
    assert res["model"] == "my_model"
    assert res["status"] == "quantized_int8"
    assert "metadata" in res
    assert res["metadata"]["quantized_modules_count"] == 1


def test_quantize_model_with_params_kwargs_dict():
    """Docstring for test_quantize_model_with_params_kwargs_dict."""
    quantize_mod.jax = MagicMock()
    quantize_mod.jnp = np

    tensor = np.array([[1.0]])
    params = {"q_proj": {"kernel": tensor}}

    res = quantize_mod.quantize_model("my_model", method="int4", params=params, quant_targets=("q_proj",))

    assert res["status"] == "quantized_int4"
    assert res["memory_reduction_factor"] == 0.75
    assert res["metadata"]["bits"] == 4


def test_quantize_model_with_params_non_dict():
    """Docstring for test_quantize_model_with_params_non_dict."""
    quantize_mod.jax = MagicMock()
    quantize_mod.jnp = np

    # Pass a non-dict for params to hit the else branch for isinstance(params, dict)
    res = quantize_mod.quantize_model("my_model", method="other", params="not_a_dict")

    assert res["status"] == "quantized_other"
    assert res["memory_reduction_factor"] == 0.7
    assert res["metadata"]["method"] == "other"
    assert res["metadata"]["memory_reduction_factor"] == 0.7


def test_quantize_model_exception_handling():
    """Docstring for test_quantize_model_exception_handling."""
    quantize_mod.jax = MagicMock()
    quantize_mod.jnp = np

    # Pass a bad tensor to cause TypeError or something in apply_aqt_quantization
    params = {
        "q_proj": {"kernel": None}  # None doesn't have .abs() or whatever jnp expects
    }

    # We'll mock apply_aqt_quantization to just raise a RuntimeError
    with patch.object(quantize_mod, "apply_aqt_quantization", side_effect=RuntimeError("mock error")):
        res = quantize_mod.quantize_model("my_model", params=params)

    assert res["status"] == "failed: mock error"
    assert "metadata" not in res  # metadata was empty when error raised
