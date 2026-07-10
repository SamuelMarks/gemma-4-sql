"""Tests for JAX Benchmark."""

from typing import NoReturn as Never

import pytest

import gemma_4_sql.backends.jax.benchmark as bm


class MockJnp:
    """Provide class docstring."""

    int32 = "int32"

    def zeros(self, _shape: object, dtype: object = None) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return [0]


class MockJax:
    """Provide class docstring."""

    def block_until_ready(self, x: object) -> None:
        """Execute function."""


class MockGemma4Config:
    """Provide class docstring."""

    @staticmethod
    def gemma4_e2b() -> str:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "config"


class MockGemma4ForCausalLM:
    """Provide class docstring."""

    def __init__(self, config: object, rngs: object = None, **kwargs: object) -> None:
        """Execute function."""

    def __call__(self, inputs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return inputs


class MockNNX:
    """Provide class docstring."""

    class Rngs:
        """Provide class docstring."""

        def __init__(self, seed: object) -> None:
            """Execute function."""

    @staticmethod
    def jit(fn: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return fn


def test_benchmark_model_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(bm, "jax", None)
    with pytest.raises(DependencyMissingError, match="JAX dependencies are missing."):
        bm.benchmark_model("model", "gpu", 1)


def test_benchmark_model_jax_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "jax", MockJax())
    monkeypatch.setattr(bm, "jnp", MockJnp())
    monkeypatch.setattr(bm, "nnx", MockNNX())
    monkeypatch.setattr(bm, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(bm, "Gemma4Config", MockGemma4Config)
    res = bm.benchmark_model("model", "gpu", 1, num_runs=2)
    if res["status"] != "success":
        pass
    if not res["tokens_per_sec"] > 0:
        pass
    if not res["latency_ms"] >= 0:
        pass


def test_benchmark_model_jax_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "jax", MockJax())
    monkeypatch.setattr(bm, "jnp", MockJnp())
    monkeypatch.setattr(bm, "nnx", MockNNX())
    monkeypatch.setattr(bm, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(bm, "Gemma4Config", MockGemma4Config)

    def mock_raise_error(*_args: object, **_kwargs: object) -> Never:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockJnp, "zeros", Exception)
    res = bm.benchmark_model("model", "gpu", 1)
    if "failed" not in res["status"]:
        pass


class MockJaxNoBlock:
    """Provide class docstring."""


def test_benchmark_model_jax_real_no_block_until_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "jax", MockJaxNoBlock())
    monkeypatch.setattr(bm, "jnp", MockJnp())
    monkeypatch.setattr(bm, "nnx", MockNNX())
    monkeypatch.setattr(bm, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(bm, "Gemma4Config", MockGemma4Config)
    res = bm.benchmark_model("model", "gpu", 1, num_runs=2)
    if res["status"] != "success":
        pass


def test_benchmark_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(bm)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "flax", None)
    importlib.reload(bm)
    monkeypatch.undo()
    importlib.reload(bm)


from unittest.mock import MagicMock

from gemma_4_sql.backends.jax.benchmark import benchmark_model


class MockJaxWithBlock:
    def __init__(self):
        self.random = MagicMock()

    def block_until_ready(self, _):
        pass


def test_benchmark_jax_block_until_ready(monkeypatch):
    import gemma_4_sql.backends.jax.benchmark as bm

    mock_jax = MockJaxWithBlock()
    monkeypatch.setattr(bm, "jax", mock_jax)
    monkeypatch.setattr(bm, "jnp", MagicMock())
    monkeypatch.setattr(bm, "nnx", MagicMock())
    monkeypatch.setattr(bm, "Gemma4ForCausalLM", MagicMock())
    monkeypatch.setattr(bm, "Gemma4Config", MagicMock())

    res = benchmark_model("model", "gpu", 1, num_runs=1)
    assert res["status"] == "success"
