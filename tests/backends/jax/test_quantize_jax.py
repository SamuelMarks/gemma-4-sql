"""Tests for JAX quantization logic."""

from __future__ import annotations

import pytest

import gemma_4_sql.backends.jax.quantize as qt
from gemma_4_sql.backends.jax.quantize import quantize_model


class MockArray:
    """Initialize class MockArray."""

    def __init__(self, ndim: int = 2) -> None:
        """Execute function."""
        self.ndim = ndim

    def __truediv__(self, other: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockArray()

    def astype(self, _dtype: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockArray()


class MockJnp:
    """Mock jnp."""

    int8 = "int8"

    def max(self, _x: object) -> float:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return 1.0

    def abs(self, _x: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockArray()

    def round(self, _x: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockArray()


class MockGemma4Config:
    """Provide class docstring."""

    @staticmethod
    def gemma4_e2b() -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "config"


class MockGemma4ForCausalLM:
    """Provide class docstring."""

    def __init__(self, config: object, rngs: object = None, **kwargs: object) -> None:
        """Execute function."""


class MockNNX:
    """Provide class docstring."""

    class Param:
        """Provide class docstring."""

        def __init__(self, value: object) -> None:
            """Execute function."""
            self.value = value

    class Rngs:
        """Provide class docstring."""

        def __init__(self, seed: int) -> None:
            """Execute function."""

    class MockGraph:
        """Provide class docstring."""

        @staticmethod
        def iter_graph(_model: object) -> list:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return [("path", MockNNX.Param(MockArray(ndim=2))), ("path2", MockNNX.Param(MockArray(ndim=1)))]

    graph = MockGraph


def test_quantize_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX quantize when missing.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(qt, "jax", None)
    with pytest.raises(DependencyMissingError, match="JAX quantization dependencies are missing."):
        quantize_model("model", "int8")


def test_quantize_jax_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX quantize real.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(qt, "jax", object())
    monkeypatch.setattr(qt, "jnp", MockJnp())
    monkeypatch.setattr(qt, "nnx", MockNNX())
    monkeypatch.setattr(qt, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(qt, "Gemma4Config", MockGemma4Config)
    res = quantize_model("model", "int8")
    if not res["status"] == "quantized_int8":
        raise AssertionError
    if not res["memory_reduction_factor"] == float("0.5"):
        raise AssertionError
    res = quantize_model("model", "awq")
    if not res["status"] == "quantized_awq":
        raise AssertionError
    res = quantize_model("model", "gptq")
    if not res["status"] == "unsupported_method_gptq":
        raise AssertionError


def test_quantize_jax_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX quantize with error.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(qt, "jax", object())
    monkeypatch.setattr(qt, "jnp", MockJnp())
    monkeypatch.setattr(qt, "nnx", MockNNX())
    monkeypatch.setattr(qt, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(qt, "Gemma4Config", MockGemma4Config)

    def mock_raise_error(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockNNX.graph, "iter_graph", mock_raise_error)
    res = quantize_model("model", "int8")
    if "failed: err" not in res["status"]:
        raise AssertionError


def test_quantize_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    mdl = __import__("gemma_4_sql.backends.jax.quantize", fromlist=[""])
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "flax", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    importlib.reload(mdl)
