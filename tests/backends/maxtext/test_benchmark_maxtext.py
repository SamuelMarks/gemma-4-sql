"""Tests for MaxText Benchmark."""

from __future__ import annotations

import typing

import gemma_4_sql.backends.maxtext.benchmark as bm
from gemma_4_sql.backends.maxtext.benchmark import benchmark_model

if typing.TYPE_CHECKING:
    import pytest


class MockJnp:
    """Provide class docstring."""

    int32 = 1

    @staticmethod
    def zeros(_shape: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return [0]


class MockJaxRandom:
    """Provide class docstring."""

    @staticmethod
    def mock_prngkey(seed: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return seed

    PRNGKey = mock_prngkey


class MockJax:
    """Provide class docstring."""

    random = MockJaxRandom()

    @staticmethod
    def jit(fn: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return fn

    def block_until_ready(self, x: object) -> None:
        """Execute function."""


class MockGemma4Model:
    """Provide class docstring."""

    def __init__(self, name: object) -> None:
        """Execute function."""

    def init(self, _rng: object, _inputs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "params"

    def apply(self, _params: object, inputs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return inputs


def test_benchmark_maxtext_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "jax", None)
    res = benchmark_model("model", "gpu", 1)
    if not res["status"] == "mocked_missing_maxtext":
        raise AssertionError


def test_benchmark_maxtext_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "jax", MockJax())
    monkeypatch.setattr(bm, "jnp", MockJnp())
    monkeypatch.setattr(bm, "Gemma4Model", MockGemma4Model)
    res = benchmark_model("model", "gpu", 1, num_runs=2, test_mode=True)
    if not res["status"] == "success":
        raise AssertionError
    if not res["tokens_per_sec"] > 0:
        raise AssertionError


def test_benchmark_maxtext_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "jax", MockJax())
    monkeypatch.setattr(bm, "jnp", MockJnp())
    monkeypatch.setattr(bm, "Gemma4Model", MockGemma4Model)

    def mock_raise_error(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockJnp, "zeros", Exception)
    res = benchmark_model("model", "gpu", 1, test_mode=True)
    if "failed" not in str(res["status"]):
        raise AssertionError


class MockModel:
    """Provide class docstring."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Execute function."""

    def init(self, *_args: object, **_kwargs: object) -> str:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "params"

    def apply(self, *_args: object, **_kwargs: object) -> str:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "out"


def test_benchmark_maxtext_real_no_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "jax", type("MockJax", (), {"random": type("MockRng", (), {"PRNGKey": lambda x: x}), "jit": lambda x: x, "distributed": type("MockDist", (), {"initialize": lambda: None})}))
    monkeypatch.setattr(bm, "jnp", type("MockJnp", (), {"int32": "int32", "zeros": lambda *args, **_kwargs: args}))
    monkeypatch.setattr(bm, "Gemma4Model", MockModel)
    res = bm.benchmark_model("model", "tpu", 1)
    if res["status"] != "success":
        raise AssertionError


def test_benchmark_maxtext_real_no_test_mode_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """

    def raise_err() -> typing.Never:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(bm, "jax", type("MockJax", (), {"random": type("MockRng", (), {"PRNGKey": lambda x: x}), "jit": lambda x: x, "distributed": type("MockDist", (), {"initialize": raise_err})}))
    monkeypatch.setattr(bm, "jnp", type("MockJnp", (), {"int32": "int32", "zeros": lambda *args, **_kwargs: args}))
    monkeypatch.setattr(bm, "Gemma4Model", MockModel)
    res = bm.benchmark_model("model", "tpu", 1)
    if res["status"] != "success":
        raise AssertionError


def test_benchmark_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    m_benchmark = __import__("gemma_4_sql.backends.maxtext.benchmark", fromlist=[""])
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(m_benchmark)
    monkeypatch.undo()
    importlib.reload(m_benchmark)
