"""Behavioral and unit tests for MLX model quantization (INT8, INT4, AWQ, GPTQ)."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, UnsupportedQuantizationMethodError
from tests.conftest import requires_mlx

pytestmark = requires_mlx


def test_mlx_quantize_imports() -> None:
    """Verify mlx.quantize gracefully sets optional imports to None when unavailable."""
    import importlib

    import gemma_4_sql.backends.mlx.quantize as quantize_module

    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None, "numpy": None}):
        importlib.reload(quantize_module)
        assert quantize_module.mlx is None
        assert quantize_module.np is None
    importlib.reload(quantize_module)


def test_calibrate_awq_scales(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify calibrate_awq_scales computes grid-searched per-channel scales and handles missing numpy."""
    import numpy as np

    import gemma_4_sql.backends.mlx.quantize as quantize_module

    with pytest.raises(ValueError, match="alpha_range"):
        quantize_module.calibrate_awq_scales(MagicMock(), MagicMock(), alpha_range=[])

    monkeypatch.setattr(quantize_module, "np", None)
    res = quantize_module.calibrate_awq_scales(MagicMock(shape=(2, 10)), MagicMock())
    assert res == [1.0] * 10

    monkeypatch.setattr(quantize_module, "np", np)
    rng = np.random.default_rng(42)
    w = rng.standard_normal(size=(10, 5))
    acts = rng.standard_normal(size=(100, 5))

    scales: Any = quantize_module.calibrate_awq_scales(w, acts)
    assert scales.shape == (5,)

    acts_1d = rng.standard_normal(size=5)
    scales2: Any = quantize_module.calibrate_awq_scales(w, acts_1d)
    assert scales2.shape == (5,)


def test_calibrate_gptq_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify calibrate_gptq_weights computes Hessian-compensated weights and handles singular matrices."""
    import numpy as np

    import gemma_4_sql.backends.mlx.quantize as quantize_module

    monkeypatch.setattr(quantize_module, "np", None)
    w_dummy: Any = "w"
    a_dummy: Any = "a"
    res = quantize_module.calibrate_gptq_weights(w_dummy, a_dummy)
    assert res == "w"

    monkeypatch.setattr(quantize_module, "np", np)
    rng = np.random.default_rng(42)
    w = rng.standard_normal(size=(10, 5))
    acts = rng.standard_normal(size=(100, 5))

    w_q: Any = quantize_module.calibrate_gptq_weights(w, acts)
    assert w_q.shape == (10, 5)

    acts_1d = rng.standard_normal(size=5)
    w_q2: Any = quantize_module.calibrate_gptq_weights(w, acts_1d)
    assert w_q2.shape == (10, 5)

    with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("singular")):
        w_q_err: Any = quantize_module.calibrate_gptq_weights(w, acts)
        assert w_q_err.shape == (10, 5)


def test_quantize_model_real_mlx() -> None:
    """Verify quantize_model quantizes a real tiny mlx.nn.Module into QuantizedLinear layers with expected scale shapes."""
    import mlx.nn as mlx_nn

    import gemma_4_sql.backends.mlx.quantize as quantize_module

    class TinyMLXModel(mlx_nn.Module):
        """Tiny MLX module with a 64x64 linear projection."""

        def __init__(self) -> None:
            """Initialize linear layer."""
            super().__init__()
            self.proj = mlx_nn.Linear(64, 64)

        def __call__(self, x: Any) -> Any:
            """Apply linear projection."""
            return self.proj(x)

    # INT8 real quantization
    import mlx.core as mx

    m_int8 = TinyMLXModel()
    assert m_int8(mx.ones((1, 64))).shape == (1, 64)
    res_int8 = quantize_module.quantize_model("tiny-mlx", "int8", model=m_int8, group_size=64)
    assert res_int8["status"] == "quantized_int8"
    assert res_int8["memory_reduction_factor"] == 0.5
    assert isinstance(m_int8.proj, mlx_nn.QuantizedLinear)
    assert m_int8.proj.scales.shape == (64, 1)

    # INT4 real quantization
    m_int4 = TinyMLXModel()
    res_int4 = quantize_module.quantize_model("tiny-mlx", "int4", model=m_int4, group_size=32)
    assert res_int4["status"] == "quantized_int4"
    assert res_int4["memory_reduction_factor"] == 0.75
    assert isinstance(m_int4.proj, mlx_nn.QuantizedLinear)
    assert m_int4.proj.scales.shape == (64, 2)

    # AWQ real quantization with calibration scales attached
    m_awq = TinyMLXModel()
    res_awq = quantize_module.quantize_model("tiny-mlx", "awq", model=m_awq, calib_data=["SELECT 1;"], group_size=32)
    assert res_awq["status"] == "quantized_awq"
    assert isinstance(m_awq.proj, mlx_nn.QuantizedLinear)
    assert hasattr(m_awq, "_awq_scales")

    # GPTQ real quantization with Hessian-compensated weights attached
    m_gptq = TinyMLXModel()
    res_gptq = quantize_module.quantize_model("tiny-mlx", "gptq", model=m_gptq, calib_data=["SELECT 1;"], group_size="32")
    assert res_gptq["status"] == "quantized_gptq"
    assert isinstance(m_gptq.proj, mlx_nn.QuantizedLinear)
    assert hasattr(m_gptq, "_gptq_compensated_weights")


def test_quantize_model_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify quantize_model error handling, unsupported methods, model loading, and immutable objects."""
    import mlx.nn as mlx_nn

    import gemma_4_sql.backends.mlx.quantize as quantize_module

    with pytest.raises(UnsupportedQuantizationMethodError):
        quantize_module.quantize_model("model", "invalid")

    class SlottedModel:
        """Model with __slots__ that rejects arbitrary attribute assignment."""

        __slots__ = ()

    with patch.object(mlx_nn, "quantize"):
        res_slot_awq = quantize_module.quantize_model("model", "awq", model=SlottedModel())
        assert res_slot_awq["status"] == "quantized_awq"
        res_slot_gptq = quantize_module.quantize_model("model", "gptq", model=SlottedModel())
        assert res_slot_gptq["status"] == "quantized_gptq"

    # Model loading via mlx_lm.load when model kwarg is omitted
    import mlx_lm

    with (
        patch.object(mlx_lm, "load", return_value=(MagicMock(), "tok")),
        patch.object(mlx_nn, "quantize"),
    ):
        res_loaded = quantize_module.quantize_model("model", "int8")
        assert res_loaded["status"] == "quantized_int8"

    with patch.object(mlx_lm, "load", side_effect=RuntimeError("load fail")):
        res_fail = quantize_module.quantize_model("model", "int8")
        assert "failed:" in res_fail["status"]

    # Missing nn.quantize attribute
    monkeypatch.delattr(mlx_nn, "quantize")
    res_no_q = quantize_module.quantize_model("model", "int8", model=MagicMock())
    assert "failed: mlx.nn.quantize is not available" in res_no_q["status"]

    # Missing mlx_lm import inside apply_fn
    with patch.dict(sys.modules, {"mlx_lm": None}), pytest.raises(DependencyMissingError):
        quantize_module.quantize_model("model", "int8", model=MagicMock())

    # Missing top-level mlx dependency
    monkeypatch.setattr(quantize_module, "mlx", None)
    with pytest.raises(DependencyMissingError):
        quantize_module.quantize_model("model", "int8")
