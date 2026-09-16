"""Tests for MLX model quantization logic (INT8, INT4, AWQ, GPTQ)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

import gemma_4_sql.backends.mlx.quantize as mquant
from gemma_4_sql.backends.mlx.quantize import (
    calibrate_awq_scales,
    calibrate_gptq_weights,
    quantize_model,
)
from gemma_4_sql.exceptions import DependencyMissingError


def test_quantize_mlx_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test quantize_model raises DependencyMissingError when mlx is missing."""
    monkeypatch.setattr(mquant, "mlx", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing."):
        quantize_model("model", "int8")


def test_quantize_mlx_unsupported_method() -> None:
    """Test quantize_model returns failure when unsupported method is passed."""
    res = quantize_model("model", "unsupported_method")
    assert "failed" in res["status"]
    assert "Unsupported quantization method" in res["status"]
    assert res["memory_reduction_factor"] == 0.0


def test_calibrate_awq_scales() -> None:
    """Test AWQ activation grid search scale calibration."""
    w = np.random.randn(16, 32).astype(np.float32)
    x = np.random.randn(8, 32).astype(np.float32)
    scales = calibrate_awq_scales(w, x)
    assert len(scales) == 32
    assert all(s > 0 for s in scales)

    with pytest.raises(ValueError, match="alpha_range must contain at least one value"):
        calibrate_awq_scales(w, x, alpha_range=())


def test_calibrate_gptq_weights() -> None:
    """Test GPTQ second-order Hessian error compensation."""
    w = np.random.randn(8, 16).astype(np.float32)
    x = np.random.randn(32, 16).astype(np.float32)
    w_gptq = calibrate_gptq_weights(w, x, damp_percent=0.05)
    assert w_gptq.shape == w.shape
    # Ensure some weights are quantized / non-identical to float originals
    assert not np.array_equal(w, w_gptq)


def test_quantize_mlx_native_int8_and_int4(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test native MLX quantization for int8, int4, awq, and gptq."""
    mock_model = MagicMock()
    mock_nn = MagicMock()
    monkeypatch.setitem(sys.modules, "mlx.nn", mock_nn)
    monkeypatch.setitem(sys.modules, "mlx", type("M", (), {"nn": mock_nn}))
    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda _n: (mock_model, None)}))

    res_int8 = quantize_model("model_8", "int8", group_size=32)
    assert res_int8["backend"] == "mlx"
    assert res_int8["status"] == "quantized_int8"
    assert res_int8["memory_reduction_factor"] == pytest.approx(0.5)
    mock_nn.quantize.assert_called_with(mock_model, group_size=32, bits=8)

    res_int4 = quantize_model("model_4", "int4")
    assert res_int4["status"] == "quantized_int4"
    assert res_int4["memory_reduction_factor"] == pytest.approx(0.75)
    mock_nn.quantize.assert_called_with(mock_model, group_size=64, bits=4)

    # AWQ calibration
    res_awq = quantize_model("model_awq", "awq")
    assert res_awq["status"] == "quantized_awq"
    assert res_awq["memory_reduction_factor"] == pytest.approx(0.75)

    # GPTQ calibration
    res_gptq = quantize_model("model_gptq", "gptq")
    assert res_gptq["status"] == "quantized_gptq"
    assert res_gptq["memory_reduction_factor"] == pytest.approx(0.75)


def test_quantize_mlx_no_silent_fallback_on_quantize_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that quantization failures do NOT fall back to fake success metrics."""
    mock_model = MagicMock()
    mock_nn = MagicMock()
    mock_nn.quantize.side_effect = RuntimeError("Incompatible tensor layout")
    monkeypatch.setitem(sys.modules, "mlx.nn", mock_nn)
    monkeypatch.setitem(sys.modules, "mlx", type("M", (), {"nn": mock_nn}))
    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda _n: mock_model}))

    res = quantize_model("faulty_model", "int8")
    assert "failed" in res["status"]
    assert "Incompatible tensor layout" in res["status"]
    assert res["memory_reduction_factor"] == 0.0


def test_calibrate_awq_and_gptq_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 1D activations, LinAlgError in pinv, np is None fallback, and model passed explicitly."""
    w = np.random.randn(8, 16).astype(np.float32)
    x_1d = np.random.randn(16).astype(np.float32)

    # 1D activations
    scales = calibrate_awq_scales(w, x_1d)
    assert len(scales) == 16
    gptq_w = calibrate_gptq_weights(w, x_1d)
    assert gptq_w.shape == w.shape

    # LinAlgError triggers pinv
    monkeypatch.setattr(np.linalg, "inv", MagicMock(side_effect=np.linalg.LinAlgError("singular")))
    gptq_pinv = calibrate_gptq_weights(w, x_1d)
    assert gptq_pinv.shape == w.shape
    monkeypatch.undo()

    # np is None fallback
    monkeypatch.setattr(mquant, "np", None)
    assert len(calibrate_awq_scales(w, x_1d)) == 16
    assert calibrate_gptq_weights(w, x_1d) is w
    monkeypatch.undo()

    # Explicit model kwarg (191->199)
    mock_model = MagicMock()
    mock_nn = MagicMock()
    monkeypatch.setitem(sys.modules, "mlx.nn", mock_nn)
    monkeypatch.setitem(sys.modules, "mlx", type("M", (), {"nn": mock_nn}))
    res = quantize_model("model", "int8", model=mock_model)
    assert res["status"] == "quantized_int8"


def test_quantize_mlx_nn_missing_quantize_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error handling when mlx.nn lacks the quantize function."""
    mock_nn_no_quant = type("MockNNNoQuant", (), {})()
    monkeypatch.setitem(sys.modules, "mlx.nn", mock_nn_no_quant)
    monkeypatch.setitem(sys.modules, "mlx", type("M", (), {"nn": mock_nn_no_quant}))
    monkeypatch.setitem(sys.modules, "mlx_lm", type("MLXLM", (), {"load": lambda _n: MagicMock()}))

    res = quantize_model("model", "int8")
    assert "failed" in res["status"]
    assert "mlx.nn.quantize is not available" in res["status"]
    assert res["memory_reduction_factor"] == 0.0


def test_quantize_mlx_missing_mlx_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test DependencyMissingError when mlx_lm is missing."""
    monkeypatch.setitem(sys.modules, "mlx_lm", None)

    with pytest.raises(DependencyMissingError, match="mlx and mlx_lm are required"):
        quantize_model("model", "int8")


def test_quantize_mlx_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test module reload when MLX is missing."""
    importlib = __import__("importlib", fromlist=[""])
    sys_mod = __import__("sys", fromlist=[""])
    mdl = __import__("gemma_4_sql.backends.mlx.quantize", fromlist=[""])
    monkeypatch.setitem(sys_mod.modules, "mlx.core", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    importlib.reload(mdl)
