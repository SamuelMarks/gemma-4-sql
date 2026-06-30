"""Tests for MaxText quantization logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gemma_4_sql.backends.maxtext.quantize as maxtext_quantize
from gemma_4_sql.backends.maxtext.quantize import quantize_model

if TYPE_CHECKING:
    import pytest


class MockJnp:
    int32 = 1

    @staticmethod
    def zeros(shape: object, **kwargs: object) -> object:
        return [0]


class MockJaxRandom:
    @staticmethod
    def PRNGKey(seed: object) -> object:
        return seed


class MockJax:
    random = MockJaxRandom()


class MockGemma4Model:
    def __init__(self, name: object) -> None:
        pass

    def init(self, rng: object, inputs: object) -> object:
        return "params"


def test_quantize_maxtext_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MaxText quantize when missing."""
    monkeypatch.setattr(maxtext_quantize, "jnp", None)
    res = quantize_model("model", "int8")
    if not res["status"] == "mocked_missing_maxtext":
        raise AssertionError
    if not res["memory_reduction_factor"] == 0.0:
        raise AssertionError


def test_quantize_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MaxText quantize."""
    monkeypatch.setattr(maxtext_quantize, "jax", MockJax())
    monkeypatch.setattr(maxtext_quantize, "jnp", MockJnp())
    monkeypatch.setattr(maxtext_quantize, "Gemma4Model", MockGemma4Model)

    res = quantize_model("model", "int8")
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["status"] == "quantized_int8":
        raise AssertionError

    res = quantize_model("model", "int4")
    if not res["status"] == "quantized_int4":
        raise AssertionError

    res = quantize_model("model", "awq")
    if not res["method"] == "awq":
        raise AssertionError
    if res["status"] not in ["quantized_awq", "mocked_missing_maxtext"]:
        raise AssertionError


def test_quantize_maxtext_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(maxtext_quantize, "jax", MockJax())
    monkeypatch.setattr(maxtext_quantize, "jnp", MockJnp())
    monkeypatch.setattr(maxtext_quantize, "Gemma4Model", MockGemma4Model)

    def raise_err(*args: object, **kwargs: object) -> object:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockJnp, "zeros", raise_err)
    res = quantize_model("model", "int8")
    if "failed" not in str(res["status"]):
        raise AssertionError
