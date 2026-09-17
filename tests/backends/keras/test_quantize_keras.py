"""Tests for Keras quantization logic and artifact export."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import gemma_4_sql.backends.keras.quantize as kr_quantize
from gemma_4_sql.backends.keras.quantize import quantize_layer_weights, quantize_model
from gemma_4_sql.exceptions import DependencyMissingError, UnsupportedQuantizationMethodError


def test_quantize_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras quantize when dependencies are missing."""
    monkeypatch.setattr(kr_quantize, "keras", None)
    with pytest.raises(DependencyMissingError, match=r"Keras dependencies are missing\."):
        quantize_model("model", "int8")


def test_quantize_keras_unsupported_methods() -> None:
    """Test Keras quantize raises UnsupportedQuantizationMethodError for unsupported methods."""
    with pytest.raises(UnsupportedQuantizationMethodError, match="Unsupported quantization method 'awq'"):
        quantize_model("model", "awq")

    with pytest.raises(UnsupportedQuantizationMethodError, match="Unsupported quantization method 'gptq'"):
        quantize_model("model", "gptq")

    with pytest.raises(UnsupportedQuantizationMethodError, match="Unsupported quantization method 'unknown'"):
        quantize_model("model", "unknown")


def test_quantize_layer_weights() -> None:
    """Test quantizing weights of a single mock Keras layer."""
    mock_weight = MagicMock()
    mock_weight.numpy.return_value = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    mock_layer = MagicMock()
    mock_layer.weights = [mock_weight]

    count_int8 = quantize_layer_weights(mock_layer, method="int8")
    assert count_int8 == 1
    mock_weight.assign.assert_called_once()

    # 1D weight (e.g. bias) is skipped
    mock_bias = MagicMock()
    mock_bias.numpy.return_value = np.array([0.5, 0.5], dtype=np.float32)
    mock_layer_1d = MagicMock()
    mock_layer_1d.weights = [mock_bias]
    assert quantize_layer_weights(mock_layer_1d, method="int4") == 0


def test_quantize_keras_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test Keras quantize with in-memory model and export artifact."""
    called_policies: list[str] = []

    class MockPolicies:
        """Mock dtype policies."""

        def set_dtype_policy(self, policy: str) -> None:
            """Set dtype policy."""
            called_policies.append(policy)

    mock_keras = type("MockKeras", (), {"dtype_policies": MockPolicies()})
    monkeypatch.setattr(kr_quantize, "keras", mock_keras)

    # In-memory model with export path
    mock_weight = MagicMock()
    mock_weight.numpy.return_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mock_layer = MagicMock()
    mock_layer.weights = [mock_weight]
    mock_model = MagicMock()
    mock_model.layers = [mock_layer]

    res = quantize_model("test_gemma", "int8", model=mock_model, export_path=str(tmp_path))
    assert res["backend"] == "keras"
    assert res["status"] == "quantized_int8"
    assert res["memory_reduction_factor"] == pytest.approx(0.5)
    assert res["quantized_layers_count"] == 1
    assert "test_gemma_int8.keras" in res["export_path"]
    mock_model.save.assert_called_once()
    assert "int8_from_float32" in called_policies


def test_quantize_keras_int4_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras int4 quantization."""
    called_policies: list[str] = []

    class MockPolicies:
        """Mock dtype policies."""

        def set_dtype_policy(self, policy: str) -> None:
            """Set dtype policy."""
            called_policies.append(policy)

    mock_keras = type("MockKeras", (), {"dtype_policies": MockPolicies()})
    monkeypatch.setattr(kr_quantize, "keras", mock_keras)

    res_int4 = quantize_model("test_model", "int4")
    assert res_int4["status"] == "quantized_int4"
    assert res_int4["memory_reduction_factor"] == pytest.approx(0.75)
    assert "int4_from_float32" in called_policies


def test_quantize_keras_config_policy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras quantize using keras.config.set_dtype_policy."""
    called_policies: list[str] = []

    class MockConfig:
        """Mock keras config."""

        def set_dtype_policy(self, policy: str) -> None:
            """Set dtype policy."""
            called_policies.append(policy)

    mock_keras = type("MockKeras", (), {"config": MockConfig()})
    monkeypatch.setattr(kr_quantize, "keras", mock_keras)

    res = quantize_model("test_model", "int8")
    assert res["status"] == "quantized_int8"
    assert "int8_from_float32" in called_policies


def test_quantize_keras_policy_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras quantize error handling when setting dtype policy fails."""

    class MockFailingPolicies:
        """Mock policies that fail."""

        def set_dtype_policy(self, _policy: str) -> None:
            """Raise RuntimeError."""
            raise RuntimeError("Dtype policy failed to apply")

    mock_keras = type("MockKeras", (), {"dtype_policies": MockFailingPolicies()})
    monkeypatch.setattr(kr_quantize, "keras", mock_keras)

    res = quantize_model("test_model", "int8")
    assert "failed: Dtype policy failed to apply" in res["status"]
    assert res["memory_reduction_factor"] == 0.0


def test_quantize_keras_no_policy_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras quantize when keras object has no policy setting functions."""
    mock_keras = type("MockKerasEmpty", (), {})
    monkeypatch.setattr(kr_quantize, "keras", mock_keras)

    res = quantize_model("test_model", "int8")
    assert res["status"] == "quantized_int8"


def test_quantize_keras_edge_cases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test np is None, weight without assign, weight raising exception, and model without save.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    import sys

    # 1. np is None
    monkeypatch.setattr(kr_quantize, "np", None)
    assert quantize_layer_weights(MagicMock()) == 0
    monkeypatch.undo()

    # 2. Weight without assign (hasattr(w, 'assign') is False)
    mock_w_no_assign = MagicMock(spec=["numpy"])
    mock_w_no_assign.numpy.return_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mock_layer_no_assign = MagicMock()
    mock_layer_no_assign.weights = [mock_w_no_assign]
    assert quantize_layer_weights(mock_layer_no_assign, method="int8") == 1

    # 3. Weight raising exception
    mock_w_err = MagicMock()
    mock_w_err.numpy.side_effect = RuntimeError("weight access error")
    mock_layer_err = MagicMock()
    mock_layer_err.weights = [mock_w_err]
    assert quantize_layer_weights(mock_layer_err, method="int8") == 0

    # 4. Model without save method and export_path provided
    mock_model_no_save = MagicMock(spec=["layers"])
    mock_model_no_save.layers = [mock_layer_no_assign]
    res_no_save = quantize_model("model_no_save", "int8", model=mock_model_no_save, export_path=str(tmp_path))
    assert res_no_save["status"] == "quantized_int8"

    # Model without export_path (covers 116->124)
    res_no_export = quantize_model("no_export", "int8", model=mock_model_no_save)
    assert res_no_export["status"] == "quantized_int8"

    # 5. Model is None and keras_nlp.models.GemmaCausalLM.from_preset raises ValueError (covers line 106)
    class MockGemmaNLP:
        """Mock keras nlp."""

        class GemmaCausalLM:
            """Mock causal LM."""

            @staticmethod
            def from_preset(name: str) -> None:
                """Raise ValueError."""
                raise ValueError("Preset not found")

    monkeypatch.setitem(sys.modules, "keras_nlp", MockGemmaNLP())
    monkeypatch.setitem(sys.modules, "keras_nlp.models", MockGemmaNLP())
    res_no_model = quantize_model("preset_fails", "int8")
    assert res_no_model["status"] == "quantized_int8"


def test_quantize_keras_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test module reload when keras is missing."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    mdl = __import__("gemma_4_sql.backends.keras.quantize", fromlist=[""])
    monkeypatch.setitem(sys.modules, "keras", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    importlib.reload(mdl)
