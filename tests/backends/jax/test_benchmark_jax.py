"""Tests for JAX Benchmark."""

from typing import NoReturn as Never

import pytest

import gemma_4_sql.backends.jax.benchmark as bm


class MockJnp:
    int32 = "int32"

    def zeros(self, shape, dtype):
        return [0]


class MockJax:
    def block_until_ready(self, x) -> None:
        pass


class MockGemma4Config:
    @staticmethod
    def gemma4_e2b() -> str:
        return "config"


class MockGemma4ForCausalLM:
    def __init__(self, config, rngs) -> None:
        pass

    def __call__(self, inputs):
        return inputs


class MockNNX:
    class Rngs:
        def __init__(self, seed) -> None:
            pass

    @staticmethod
    def jit(fn):
        return fn


def test_benchmark_model_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "jax", None)
    res = bm.benchmark_model("model", "gpu", 1)
    if not res["status"] == "mocked_missing_jax":
        raise AssertionError


def test_benchmark_model_jax_real(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "jax", MockJax())
    monkeypatch.setattr(bm, "jnp", MockJnp())
    monkeypatch.setattr(bm, "nnx", MockNNX())
    monkeypatch.setattr(bm, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(bm, "Gemma4Config", MockGemma4Config)

    res = bm.benchmark_model("model", "gpu", 1, num_runs=2)
    if not res["status"] == "success":
        raise AssertionError
    if not res["tokens_per_sec"] > 0:
        raise AssertionError
    if not res["latency_ms"] >= 0:
        raise AssertionError


def test_benchmark_model_jax_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "jax", MockJax())
    monkeypatch.setattr(bm, "jnp", MockJnp())
    monkeypatch.setattr(bm, "nnx", MockNNX())
    monkeypatch.setattr(bm, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(bm, "Gemma4Config", MockGemma4Config)

    def raise_error(*args, **kwargs) -> Never:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockJnp, "zeros", raise_error)

    res = bm.benchmark_model("model", "gpu", 1)
    if "failed" not in res["status"]:
        raise AssertionError


def test_benchmark_model_jax_real_no_block_until_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockJaxNoBlock:
        pass

    monkeypatch.setattr(bm, "jax", MockJaxNoBlock())
    monkeypatch.setattr(bm, "jnp", MockJnp())
    monkeypatch.setattr(bm, "nnx", MockNNX())
    monkeypatch.setattr(bm, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(bm, "Gemma4Config", MockGemma4Config)

    res = bm.benchmark_model("model", "gpu", 1, num_runs=2)
    assert res["status"] == "success"


def test_benchmark_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    # Mock jax import failure
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(bm)

    # Mock flax.nnx import failure
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "flax", None)
    importlib.reload(bm)

    # Restore original to not break other tests
    monkeypatch.undo()
    importlib.reload(bm)
