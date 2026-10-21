"""Unit and behavioral tests for quantize keras."""

from unittest.mock import MagicMock

from tests.conftest import requires_keras

pytestmark = requires_keras

import numpy as np
import pytest

import gemma_4_sql.backends.keras.quantize as q


def test_quantize_layer_weights(monkeypatch):
    """Execute and verify `test_quantize_layer_weights` behavior and invariants."""
    mock_layer = MagicMock()
    mock_layer.name = "dense"

    # Missing np
    monkeypatch.setattr(q, "np", None)
    assert q.quantize_layer_weights(mock_layer, "int8") == 0

    monkeypatch.setattr(q, "np", np)

    # Needs np array ndim >= 2
    mock_w = MagicMock()
    mock_w.name = "kernel"
    mock_w.numpy.return_value = np.array([[1.0, -1.0], [2.0, -2.0]])
    mock_layer.weights = [mock_w]

    # int8
    res = q.quantize_layer_weights(mock_layer, "int8")
    assert res > 0

    # int4 and zero max
    mock_layer3 = MagicMock()
    mock_layer3.name = "dense"
    mock_w3 = MagicMock()
    mock_w3.name = "kernel"
    mock_w3.numpy.return_value = np.array([[0.0, 0.0]])  # ndim=2, zero max
    del mock_w3.assign  # make hasattr(w, "assign") False
    mock_layer3.weights = [mock_w3]
    res3 = q.quantize_layer_weights(mock_layer3, "int4")
    assert res3 > 0

    # ndim < 2
    mock_layer_ndim1 = MagicMock()
    mock_w_ndim1 = MagicMock()
    mock_w_ndim1.numpy.return_value = np.array([1.0, 2.0])
    mock_layer_ndim1.weights = [mock_w_ndim1]
    res_ndim1 = q.quantize_layer_weights(mock_layer_ndim1, "int8")
    assert res_ndim1 == 0

    # runtime error
    mock_layer4 = MagicMock()
    mock_w4 = MagicMock()
    mock_w4.numpy.side_effect = RuntimeError("sim")
    mock_layer4.weights = [mock_w4]
    assert q.quantize_layer_weights(mock_layer4, "int8") == 0


def test_quantize_model(monkeypatch, tmp_path):
    """Execute and verify `test_quantize_model` behavior and invariants."""
    from gemma_4_sql.exceptions import DependencyMissingError, UnsupportedQuantizationMethodError

    # 1. Missing keras
    monkeypatch.setattr(q, "keras", None)
    with pytest.raises(DependencyMissingError):
        q.quantize_model("m", "int8")

    mock_keras = MagicMock()
    monkeypatch.setattr(q, "keras", mock_keras)

    # 2. Unsupported method
    with pytest.raises(UnsupportedQuantizationMethodError):
        q.quantize_model("m", "unknown")

    # 3. Model passed in kwargs, success, export_path
    mock_model = MagicMock()
    mock_layer = MagicMock()
    mock_model.layers = [mock_layer]

    monkeypatch.setattr(q, "quantize_layer_weights", MagicMock(return_value=1))

    export_dir = tmp_path / "export"
    res = q.quantize_model("m", "int8", model=mock_model, export_path=str(export_dir))
    assert res["status"] == "quantized_int8"
    assert "export_path" in res
    assert res["quantized_layers_count"] == 1

    # 4. Keras config.set_dtype_policy mock
    del mock_keras.dtype_policies
    mock_keras.config.set_dtype_policy = MagicMock()
    q.quantize_model("m", "int4", model=mock_model)

    # 4.b Keras without dtype_policies and without config
    del mock_keras.config
    q.quantize_model("m", "int4", model=mock_model)

    # 5. Load model from preset
    mock_gemma_cls = MagicMock()
    mock_gemma_cls.from_preset.return_value = mock_model
    monkeypatch.setattr(q, "optional_attr", lambda *_a, **_k: mock_gemma_cls)

    res = q.quantize_model("m", "int8")
    assert res["status"] == "quantized_int8"

    # 6. Load model from preset error / None
    monkeypatch.setattr(q, "optional_attr", lambda *_a, **_k: None)
    res_none = q.quantize_model("m", "int8")
    assert res_none["status"] == "quantized_int8"

    mock_gemma_cls.from_preset.side_effect = ValueError("sim")
    monkeypatch.setattr(q, "optional_attr", lambda *_a, **_k: mock_gemma_cls)
    res = q.quantize_model("m", "int8")
    assert res["status"] == "quantized_int8"  # Doesn't fail, model is just None

    # 7. Model runtime error
    mock_model.save.side_effect = RuntimeError("sim")
    res = q.quantize_model("m", "int8", model=mock_model, export_path=str(export_dir))
    assert "failed" in res["status"]
    assert res["memory_reduction_factor"] == 0.0

    # 8. Model without save
    mock_model_nosave = MagicMock()
    del mock_model_nosave.save
    mock_model_nosave.layers = [mock_layer]
    res = q.quantize_model("m", "int8", model=mock_model_nosave, export_path=str(export_dir))
    assert res["status"] == "quantized_int8"
