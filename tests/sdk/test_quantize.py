"""Tests for Quantization SDK module."""

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.quantize import quantize_model


def test_quantize_jax() -> None:
    """Initialize function test_quantize_jax.

    Raises:
        AssertionError: Description.

    """
    res = quantize_model("model1", "int8", backend="jax")
    assert res["backend"] == "jax"
    assert res["model"] == "model1"
    assert res["method"] == "int8"


def test_quantize_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function test_quantize_pytorch.

    Raises:
        AssertionError: Description.

    """
    import gemma_4_sql.backends.pytorch.quantize as pt_q

    monkeypatch.setattr(pt_q, "torch", None)
    with pytest.raises(DependencyMissingError):
        quantize_model("model2", "int4", backend="pytorch")


def test_quantize_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras quantization through SDK."""
    from gemma_4_sql.exceptions import UnsupportedQuantizationMethodError

    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    keras_agent = get_backend("keras")
    monkeypatch.setattr(keras_agent, "quantize_model", lambda m, method, **k: {"backend": "keras", "model": m, "method": method, "status": f"quantized_{method}"})

    res = quantize_model("model3", "int8", backend="keras")
    assert res["backend"] == "keras"
    assert res["model"] == "model3"
    assert res["status"] == "quantized_int8"

    # For the unsupported method error, we need to mock it raising it
    def raise_unsupported(*args, **kwargs):
        raise UnsupportedQuantizationMethodError("awq is not supported")

    monkeypatch.setattr(keras_agent, "quantize_model", raise_unsupported)

    with pytest.raises(UnsupportedQuantizationMethodError):
        quantize_model("model3", "awq", backend="keras")


def test_quantize_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function test_quantize_maxtext."""
    import gemma_4_sql.backends.maxtext.quantize as mx_q

    monkeypatch.setattr(mx_q, "jax", None)
    with pytest.raises(DependencyMissingError):
        quantize_model("model4", "gguf", backend="maxtext")


def test_quantize_invalid() -> None:
    """Initialize function test_quantize_invalid."""
    with pytest.raises(ValueError, match=r".*"):
        quantize_model("model", "int8", backend="unknown")
