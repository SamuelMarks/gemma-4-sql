"""Behavioral unit tests for the Quantization SDK module."""

from __future__ import annotations

from pathlib import Path

import keras
import pytest
from keras import ops

from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError, UnsupportedQuantizationMethodError
from gemma_4_sql.sdk.quantize import quantize_model


def test_quantize_jax() -> None:
    """Verify `quantize_model` executes INT8 weight quantization on the JAX backend."""
    res = quantize_model("model1", "int8", backend="jax")
    assert res["backend"] == "jax"
    assert res["model"] == "model1"
    assert res["method"] == "int8"
    assert res["memory_reduction_factor"] > 0.0


def test_quantize_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `quantize_model` raises `DependencyMissingError` when PyTorch is unavailable."""
    import gemma_4_sql.backends.pytorch.quantize as pt_q

    monkeypatch.setattr(pt_q, "torch", None)
    with pytest.raises(DependencyMissingError):
        quantize_model("model2", "int4", backend="pytorch")


def test_quantize_keras(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify `quantize_model` executes Keras INT8 quantization and rejects unsupported methods."""
    import gemma_4_sql.backends.keras.quantize as kq

    class DummyLayer:
        """Minimal layer stub with 2D Keras tensor weights."""

        def __init__(self) -> None:
            """Initialize layer stub."""
            self.weights = [ops.reshape(ops.linspace(1.0, 4.0, 4), (2, 2))]

    class DummyKerasModel:
        """Minimal Keras model stub with quantizable layers."""

        def __init__(self) -> None:
            """Initialize model stub with two layers."""
            self.layers = [DummyLayer(), DummyLayer()]
            self.saved_path: str | None = None

        def save(self, path: str | Path) -> None:
            """Write dummy file to verify export_path persistence."""
            p = Path(path)
            p.write_bytes(b"keras_quantized")
            self.saved_path = str(p)

    dummy_model = DummyKerasModel()

    class DummyGemmaCausalLM:
        """Stub for GemmaCausalLM.from_preset."""

        @staticmethod
        def from_preset(_name: str) -> DummyKerasModel:
            """Return the DummyKerasModel instance."""
            return dummy_model

    monkeypatch.setattr(kq, "keras", keras)
    monkeypatch.setattr(kq, "ops", ops)
    monkeypatch.setattr(kq, "optional_attr", lambda *_a: DummyGemmaCausalLM)  # pyright: ignore[reportUnknownLambdaType]
    export_dest = tmp_path / "q_dir"

    res = quantize_model("model3", "int8", backend="keras", export_path=str(export_dest))
    assert res["backend"] == "keras"
    assert res["model"] == "model3"
    assert res["status"] == "quantized_int8"
    assert res["quantized_layers_count"] == 2
    assert Path(str(res["export_path"])).exists()

    with pytest.raises(UnsupportedQuantizationMethodError):
        quantize_model("model3", "awq", backend="keras", strict_method=True)


def test_quantize_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `quantize_model` raises `DependencyMissingError` when MaxText dependencies are missing."""
    import gemma_4_sql.backends.maxtext.quantize as mx_q

    monkeypatch.setattr(mx_q, "jax", None)
    with pytest.raises(DependencyMissingError):
        quantize_model("model4", "gguf", backend="maxtext")


def test_quantize_invalid() -> None:
    """Verify `quantize_model` raises `ConfigurationError` (a `ValueError` subclass) on an unknown backend."""
    with pytest.raises(ConfigurationError, match="Unknown backend: unknown"):
        quantize_model("model", "int8", backend="unknown")
