"""Unit and behavioral tests for Keras model quantization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import keras
import pytest
from keras import ops

import gemma_4_sql.backends.keras.quantize as q
from gemma_4_sql.exceptions import DependencyMissingError, UnsupportedQuantizationMethodError
from tests.conftest import requires_keras

pytestmark = requires_keras


def test_quantize_layer_weights_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify quantize_layer_weights quantizes real Keras Dense 2D kernels in-place for int8 and int4."""
    dense_int8 = keras.layers.Dense(4, use_bias=True, name="dense_int8")
    _ = dense_int8(ops.ones((2, 4), dtype="float32"))
    orig_w = ops.reshape(ops.linspace(-2.0, 2.0, 16), (4, 4))
    assert isinstance(dense_int8.kernel, keras.Variable)
    dense_int8.kernel.assign(orig_w)

    count_int8 = q.quantize_layer_weights(dense_int8, "int8")
    assert count_int8 == 1
    q_w8 = ops.stop_gradient(dense_int8.kernel)
    assert bool(ops.all(ops.isclose(q_w8, orig_w, atol=0.05)))
    assert not bool(ops.all(ops.equal(q_w8, orig_w)))

    dense_int4 = keras.layers.Dense(4, use_bias=False, name="dense_int4")
    _ = dense_int4(ops.ones((2, 4), dtype="float32"))
    assert isinstance(dense_int4.kernel, keras.Variable)
    dense_int4.kernel.assign(orig_w)
    count_int4 = q.quantize_layer_weights(dense_int4, "int4")
    assert count_int4 == 1
    q_w4 = ops.stop_gradient(dense_int4.kernel)
    assert bool(ops.all(ops.isclose(q_w4, orig_w, atol=0.35)))

    # All-zero 2D weight matrix and weight without .assign()
    mock_layer_zero = MagicMock()
    mock_layer_zero.weights = [ops.zeros((2, 2), dtype="float32")]
    assert q.quantize_layer_weights(mock_layer_zero, "int4") == 1

    # Runtime error inside weight extraction is caught gracefully
    mock_layer_err = MagicMock()
    mock_layer_err.weights = ["invalid_weight"]
    assert q.quantize_layer_weights(mock_layer_err, "int8") == 0

    # Missing keras.ops returns 0
    monkeypatch.setattr(q, "ops", None)
    assert q.quantize_layer_weights(dense_int8, "int8") == 0


def test_quantize_model_real(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify quantize_model quantizes a real Keras Sequential model and saves weights."""
    inp = keras.Input(shape=(4,))
    dense_proj = keras.layers.Dense(4, name="proj")
    out = dense_proj(inp)
    real_model = keras.Model(inputs=inp, outputs=out)
    orig_w = ops.reshape(ops.linspace(-1.5, 1.5, 16), (4, 4))
    assert isinstance(dense_proj.kernel, keras.Variable)
    dense_proj.kernel.assign(orig_w)

    export_dir = tmp_path / "export"
    res = q.quantize_model("m", "int8", model=real_model, export_path=str(export_dir))
    assert res["status"] == "quantized_int8"
    assert res["quantized_layers_count"] == 1
    assert Path(str(res["export_path"])).exists()
    assert not bool(ops.all(ops.equal(ops.stop_gradient(dense_proj.kernel), orig_w)))

    # Unsupported method raises UnsupportedQuantizationMethodError
    with pytest.raises(UnsupportedQuantizationMethodError):
        q.quantize_model("m", "unknown")

    # Preset loading fallback and error paths
    mock_gemma_cls = MagicMock()
    mock_gemma_cls.from_preset.return_value = real_model
    monkeypatch.setattr(q, "optional_attr", lambda *_a, **_k: mock_gemma_cls)  # pyright: ignore[reportUnknownLambdaType]
    res_preset = q.quantize_model("m", "int4")
    assert res_preset["status"] == "quantized_int4"

    mock_gemma_cls.from_preset.return_value = None
    assert q.quantize_model("m", "int8")["status"] == "quantized_int8"

    monkeypatch.setattr(q, "optional_attr", lambda *_a, **_k: None)  # pyright: ignore[reportUnknownLambdaType]
    assert q.quantize_model("m", "int8")["status"] == "quantized_int8"

    mock_gemma_cls.from_preset.side_effect = ValueError("sim")
    monkeypatch.setattr(q, "optional_attr", lambda *_a, **_k: mock_gemma_cls)  # pyright: ignore[reportUnknownLambdaType]
    assert q.quantize_model("m", "int8")["status"] == "quantized_int8"

    # Keras dtype policy fallback branches
    mock_keras = MagicMock()
    del mock_keras.config
    mock_keras.dtype_policies.set_dtype_policy = MagicMock()
    monkeypatch.setattr(q, "keras", mock_keras)
    q.quantize_model("m", "int4", model=real_model)

    del mock_keras.dtype_policies
    q.quantize_model("m", "int4", model=real_model)

    # Save failure handling
    bad_save_model = MagicMock()
    bad_save_model.layers = []
    bad_save_model.save.side_effect = RuntimeError("save failed")
    res_fail = q.quantize_model("m", "int8", model=bad_save_model, export_path=str(export_dir))
    assert "failed" in str(res_fail["status"])
    assert res_fail["memory_reduction_factor"] == 0.0

    # Model without save attribute
    nosave_model = MagicMock()
    del nosave_model.save
    nosave_model.layers = []
    assert q.quantize_model("m", "int8", model=nosave_model, export_path=str(export_dir))["status"] == "quantized_int8"

    # Missing keras raises DependencyMissingError
    keras.config.set_dtype_policy("float32")
    monkeypatch.setattr(q, "keras", None)
    with pytest.raises(DependencyMissingError):
        q.quantize_model("m", "int8")
