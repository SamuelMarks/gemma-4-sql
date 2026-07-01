"""Tests for JAX quantization logic."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import gemma_4_sql.backends.jax.quantize as qt
from gemma_4_sql.backends.jax.quantize import quantize_model

if TYPE_CHECKING:
    import pytest


class MockArray:
    """Initialize class MockArray."""

    def __init__(self: typing.Any, ndim: int = 2) -> None:
        self.ndim = ndim

    def __truediv__(self: typing.Any, other: object) -> object:
        return MockArray()

    def astype(self: typing.Any, _dtype: object) -> object:
        return MockArray()


class MockJnp:
    """Mock jnp."""

    int8 = "int8"

    def max(self: typing.Any, _x: object) -> float:
        return 1.0

    def abs(self: typing.Any, _x: object) -> object:
        return MockArray()

    def round(self: typing.Any, _x: object) -> object:
        return MockArray()


class MockGemma4Config:
    @staticmethod
    def gemma4_e2b() -> object:
        return "config"


class MockGemma4ForCausalLM:
    def __init__(self: typing.Any, config: object, rngs: object) -> None:
        pass


class MockNNX:
    class Param:
        def __init__(self: typing.Any, value: object) -> None:
            self.value = value

    class Rngs:
        def __init__(self: typing.Any, seed: int) -> None:
            pass

    class graph:
        @staticmethod
        def iter_graph(_model: object) -> list:
            return [("path", MockNNX.Param(MockArray(ndim=2))), ("path2", MockNNX.Param(MockArray(ndim=1)))]


def test_quantize_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX quantize when missing."""
    monkeypatch.setattr(qt, "jax", None)
    res = quantize_model("model", "int8")
    if not res["status"] == "mocked_missing_jax":
        raise AssertionError
    if not res["memory_reduction_factor"] == 0.0:
        raise AssertionError


def test_quantize_jax_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX quantize real."""
    monkeypatch.setattr(qt, "jax", object())
    monkeypatch.setattr(qt, "jnp", MockJnp())
    monkeypatch.setattr(qt, "nnx", MockNNX())
    monkeypatch.setattr(qt, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(qt, "Gemma4Config", MockGemma4Config)

    res = quantize_model("model", "int8")
    if not res["status"] == "quantized_int8":
        raise AssertionError
    if not res["memory_reduction_factor"] == 0.5:
        raise AssertionError

    res = quantize_model("model", "awq")
    if not res["status"] == "quantized_awq":
        raise AssertionError

    res = quantize_model("model", "gptq")
    if not res["status"] == "unsupported_method_gptq":
        raise AssertionError


def test_quantize_jax_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX quantize with error."""
    monkeypatch.setattr(qt, "jax", object())
    monkeypatch.setattr(qt, "jnp", MockJnp())
    monkeypatch.setattr(qt, "nnx", MockNNX())
    monkeypatch.setattr(qt, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(qt, "Gemma4Config", MockGemma4Config)

    def raise_error(*_args: object, **_kwargs: object) -> object:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockNNX.graph, "iter_graph", raise_error)

    res = quantize_model("model", "int8")
    if "failed: err" not in res["status"]:
        raise AssertionError


def test_quantize_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    import gemma_4_sql.backends.jax.quantize as mdl

    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(mdl)

    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "flax", None)
    importlib.reload(mdl)

    monkeypatch.undo()
    importlib.reload(mdl)
