"""Tests for MaxText Benchmark."""

from __future__ import annotations

import typing

import gemma_4_sql.backends.maxtext.benchmark as bm
from gemma_4_sql.backends.maxtext.benchmark import benchmark_model

if typing.TYPE_CHECKING:
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

    @staticmethod
    def jit(fn: object) -> object:
        return fn

    def block_until_ready(self, x: object) -> None:
        pass


class MockGemma4Model:
    def __init__(self, name: object) -> None:
        pass

    def init(self, rng: object, inputs: object) -> object:
        return "params"

    def apply(self, params: object, inputs: object) -> object:
        return inputs


def test_benchmark_maxtext_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "jax", None)
    res = benchmark_model("model", "gpu", 1)
    if not res["status"] == "mocked_missing_maxtext":
        raise AssertionError


def test_benchmark_maxtext_real(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "jax", MockJax())
    monkeypatch.setattr(bm, "jnp", MockJnp())
    monkeypatch.setattr(bm, "Gemma4Model", MockGemma4Model)

    res = benchmark_model("model", "gpu", 1, num_runs=2, test_mode=True)
    if not res["status"] == "success":
        raise AssertionError
    if not res["tokens_per_sec"] > 0:
        raise AssertionError


def test_benchmark_maxtext_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "jax", MockJax())
    monkeypatch.setattr(bm, "jnp", MockJnp())
    monkeypatch.setattr(bm, "Gemma4Model", MockGemma4Model)

    def raise_error(*args: object, **kwargs: object) -> object:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockJnp, "zeros", raise_error)

    res = benchmark_model("model", "gpu", 1, test_mode=True)
    if "failed" not in str(res["status"]):
        raise AssertionError


def test_benchmark_maxtext_real_no_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "jax", type("MockJax", (), {"random": type("MockRng", (), {"PRNGKey": lambda x: x}), "jit": lambda x: x, "distributed": type("MockDist", (), {"initialize": lambda: None})}))
    monkeypatch.setattr(bm, "jnp", type("MockJnp", (), {"int32": "int32", "zeros": lambda *args, **kwargs: args}))

    class MockModel:
        def __init__(self, *args, **kwargs):
            pass

        def init(self, *args, **kwargs):
            return "params"

        def apply(self, *args, **kwargs):
            return "out"

    monkeypatch.setattr(bm, "Gemma4Model", MockModel)
    res = bm.benchmark_model("model", "tpu", 1)
    assert res["status"] == "success"


def test_benchmark_maxtext_real_no_test_mode_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_err():
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(bm, "jax", type("MockJax", (), {"random": type("MockRng", (), {"PRNGKey": lambda x: x}), "jit": lambda x: x, "distributed": type("MockDist", (), {"initialize": raise_err})}))
    monkeypatch.setattr(bm, "jnp", type("MockJnp", (), {"int32": "int32", "zeros": lambda *args, **kwargs: args}))

    class MockModel:
        def __init__(self, *args, **kwargs):
            pass

        def init(self, *args, **kwargs):
            return "params"

        def apply(self, *args, **kwargs):
            return "out"

    monkeypatch.setattr(bm, "Gemma4Model", MockModel)
    res = bm.benchmark_model("model", "tpu", 1)
    assert res["status"] == "success"


def test_benchmark_imports_fail(monkeypatch: pytest.MonkeyPatch):
    import importlib
    import sys

    import gemma_4_sql.backends.maxtext.benchmark as m_benchmark

    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(m_benchmark)
    monkeypatch.undo()
    importlib.reload(m_benchmark)
