"""Behavioral unit tests for the Quantization SDK module."""

from __future__ import annotations

from typing import Any

import pytest

from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError, UnsupportedQuantizationMethodError
from gemma_4_sql.sdk.quantize import quantize_model
from gemma_4_sql.sdk.registry import get_backend


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


def test_quantize_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `quantize_model` forwards method and extra `**kwargs` to the Keras backend."""
    keras_agent = get_backend("keras")
    captured_kw: dict[str, Any] = {}

    def _spy_quantize(m: str, method: str, **kw: Any) -> dict[str, Any]:
        """Record forwarded quantization kwargs."""
        captured_kw.update(kw)
        return {
            "backend": "keras",
            "model": m,
            "method": method,
            "status": f"quantized_{method}",
            "memory_reduction_factor": 2.0,
        }

    monkeypatch.setattr(keras_agent, "quantize_model", _spy_quantize)

    res = quantize_model("model3", "int8", backend="keras", export_path="/tmp/q.keras")
    assert res["backend"] == "keras"
    assert res["model"] == "model3"
    assert res["status"] == "quantized_int8"
    assert captured_kw["export_path"] == "/tmp/q.keras"

    def raise_unsupported(*_args: Any, **_kwargs: Any) -> None:
        """Raise UnsupportedQuantizationMethodError."""
        raise UnsupportedQuantizationMethodError("awq is not supported")

    monkeypatch.setattr(keras_agent, "quantize_model", raise_unsupported)

    with pytest.raises(UnsupportedQuantizationMethodError):
        quantize_model("model3", "awq", backend="keras")


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
