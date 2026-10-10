"""Module docstring."""

from unittest.mock import MagicMock

import numpy as np
import pytest

import gemma_4_sql.backends.keras.quantize as q


def test_module_load_import_error(monkeypatch):
    """Docstring for test_module_load_import_error."""
    with open(q.__file__) as f:
        code = f.read()

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name in ("keras", "numpy"):
            raise ImportError("simulated missing import")
        return orig_import(name, *args, **kwargs)

    namespace = {"__name__": "mock_quantize", "__builtins__": dict(builtins.__dict__)}
    namespace["__builtins__"]["__import__"] = mock_import

    exec(code, namespace)  # noqa: S102

    assert namespace.get("keras") is None
    assert namespace.get("np") is None


def test_quantize_layer_weights(monkeypatch):
    """Docstring for test_quantize_layer_weights."""
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
    """Docstring for test_quantize_model."""
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
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "keras_nlp.models":
            mock_models = MagicMock()
            mock_models.GemmaCausalLM.from_preset.return_value = mock_model
            return mock_models
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    res = q.quantize_model("m", "int8")
    assert res["status"] == "quantized_int8"

    # 6. Load model from preset error
    def mock_import_err(*args, **kwargs):
        """Docstring for mock_import_err."""
        raise ValueError("sim")

    monkeypatch.setattr(builtins, "__import__", mock_import_err)

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


def test_module_load_import_error_coverage(monkeypatch):
    """Docstring for test_module_load_import_error_coverage."""
    import importlib
    import sys

    import gemma_4_sql.backends.keras.quantize as q

    monkeypatch.setitem(sys.modules, "keras", None)
    importlib.reload(q)
    assert q.keras is None
    assert q.np is None
    monkeypatch.undo()
    importlib.reload(q)
