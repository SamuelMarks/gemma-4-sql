"""
Tests for SDK quantize module.
"""

from __future__ import annotations

import pytest

from gemma_4_sql.sdk.quantize import quantize_model


def test_quantize_pytorch() -> None:
    res = quantize_model("model1", "int8", backend="pytorch")
    assert res["backend"] == "pytorch"
    assert res["model"] == "model1"
    assert res["method"] == "int8"


def test_quantize_jax() -> None:
    res = quantize_model("model2", "awq", backend="jax")
    assert res["backend"] == "jax"
    assert res["model"] == "model2"
    assert res["method"] == "awq"


def test_quantize_keras() -> None:
    res = quantize_model("model3", "gptq", backend="keras")
    assert res["backend"] == "keras"
    assert res["model"] == "model3"
    assert res["method"] == "gptq"


def test_quantize_maxtext() -> None:
    res = quantize_model("model4", "gguf", backend="maxtext")
    assert res["backend"] == "maxtext"
    assert res["model"] == "model4"
    assert res["method"] == "gguf"


def test_quantize_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown backend: missing"):
        quantize_model("my-model", "int8", backend="missing")
