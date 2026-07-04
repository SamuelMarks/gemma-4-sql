# Copyright 2024
"""Tests for JAX DPO logic."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NoReturn as Never

import gemma_4_sql.backends.jax.dpo as tr
from gemma_4_sql.backends.jax.dpo import dpo_loss, run_dpo
from gemma_4_sql.type_hints import DPOConfig

if TYPE_CHECKING:
    import pytest


class MockArray:
    """Initialize class MockArray."""

    def __sub__(self: object, other: object) -> MockArray:
        """Initialize function __sub__.

        Returns:
            object: Description of return.

        """
        return MockArray()

    def __mul__(self: object, other: object) -> MockArray:
        """Initialize function __mul__.

        Returns:
            object: Description of return.

        """
        return MockArray()

    def __rmul__(self: object, other: object) -> MockArray:
        """Initialize function __rmul__.

        Returns:
            object: Description of return.

        """
        return MockArray()

    def __neg__(self: object) -> MockArray:
        """Initialize function __neg__.

        Returns:
            object: Description of return.

        """
        return MockArray()

    def item(self: object) -> float:
        """Initialize function item.

        Returns:
            object: Description of return.

        """
        return 0.42


class MockJnp:
    """Initialize class MockJnp."""

    int32 = 1

    def array(self: object, *_args: object, **_kwargs: object) -> MockArray:
        """Initialize function array.

        Returns:
            object: Description of return.

        """
        return MockArray()

    def zeros(self: object, *_args: object, **_kwargs: object) -> MockArray:
        """Initialize function zeros.

        Returns:
            object: Description of return.

        """
        return MockArray()

    def mean(self: object, _x: object) -> MockArray:
        """Initialize function mean.

        Returns:
            object: Description of return.

        """
        return MockArray()

    def sum(self: object, _x: object, **_kwargs: object) -> MockArray:
        """Initialize function sum.

        Returns:
            object: Description of return.

        """
        return MockArray()


class MockJnn:
    """Initialize class MockJnn."""

    def log_sigmoid(self: object, _x: object) -> MockArray:
        """Initialize function log_sigmoid.

        Returns:
            object: Description of return.

        """
        return MockArray()


class MockOptax:
    """Initialize class MockOptax."""

    def adamw(self: object, _lr: object) -> object:
        """Initialize function adamw.

        Returns:
            object: Description of return.

        """
        return "opt_state"


class MockGemma4Config:
    """Initialize class MockGemma4Config."""

    @staticmethod
    def gemma4_e2b() -> object:
        """Initialize function gemma4_e2b.

        Returns:
            object: Description of return.

        """
        return "mock_config"


class MockGemma4ForCausalLM:
    """Initialize class MockGemma4ForCausalLM."""

    def __init__(self, config: object, rngs: object = None, dtype: object = None) -> None:
        """Initialize function __init__."""
        self.config = config

    def __call__(self, _inputs: object) -> object:
        """Initialize function __call__.

        Returns:
            object: Description of return.

        """
        return MockArray()


class MockNNXOptimizer:
    """Initialize class MockNNXOptimizer."""

    def __init__(self, model: object, optax_optimizer: object) -> None:
        """Initialize function __init__."""
        self.model = model
        self.optax_optimizer = optax_optimizer

    def update(self, grads: object) -> object:
        """Initialize function update."""


class MockNNX:
    """Initialize class MockNNX."""

    class Rngs:
        """Initialize class Rngs."""

        def __init__(self, seed: object) -> None:
            """Initialize function __init__."""
            self.seed = seed

    @staticmethod
    def jit(fn: object) -> object:
        """Initialize function jit.

        Returns:
            object: Description of return.

        """
        return fn

    @staticmethod
    def value_and_grad(fn: object) -> object:
        """Initialize function value_and_grad.

        Returns:
            object: Description of return.

        """

        def wrapper(*_args: object, **_kwargs: object) -> object:
            """Initialize function wrapper.

            Returns:
                object: Description of return.

            """
            _ = fn(*_args, **_kwargs)
            return (MockArray(), "grads")

        return wrapper

    Optimizer = MockNNXOptimizer


def test_run_dpo_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX DPO when missing.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(tr, "jnp", None)
    monkeypatch.setattr(tr, "jnn", None)
    monkeypatch.setattr(tr, "jax", None)
    res = run_dpo(DPOConfig(model_name="model", dataset="data"))
    if not res["status"] == "mocked_missing_jax":
        raise AssertionError
    (loss, ch_r, re_r) = dpo_loss(None, None, None, None)
    if not loss == 0.0:
        raise AssertionError
    if not ch_r == 0.0:
        raise AssertionError
    if not re_r == 0.0:
        raise AssertionError


def test_run_dpo_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX DPO.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(tr, "jnp", MockJnp())
    monkeypatch.setattr(tr, "jnn", MockJnn())
    monkeypatch.setattr(tr, "jax", object())
    monkeypatch.setattr(tr, "optax", MockOptax())
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(tr, "Gemma4Config", MockGemma4Config)
    monkeypatch.setattr(tr, "nnx", MockNNX())

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return {"loader": [{"chosen_inputs": MockArray(), "chosen_labels": MockArray(), "rejected_inputs": MockArray(), "rejected_labels": MockArray()}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    res = run_dpo(DPOConfig(model_name="model", dataset="data"))
    if not res["backend"] == "jax":
        raise AssertionError


def test_run_dpo_jax_no_loader_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX DPO without dataloader.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(tr, "jnp", MockJnp())
    monkeypatch.setattr(tr, "jnn", MockJnn())
    monkeypatch.setattr(tr, "jax", object())
    monkeypatch.setattr(tr, "optax", MockOptax())
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(tr, "Gemma4Config", MockGemma4Config)
    monkeypatch.setattr(tr, "nnx", MockNNX())

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return {"loader": None}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    res = run_dpo(DPOConfig(model_name="model", dataset="data"))
    if not res["backend"] == "jax":
        raise AssertionError


def test_run_dpo_jax_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX DPO with error.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(tr, "jnp", MockJnp())
    monkeypatch.setattr(tr, "jnn", MockJnn())
    monkeypatch.setattr(tr, "jax", object())
    monkeypatch.setattr(tr, "optax", MockOptax())
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(tr, "Gemma4Config", MockGemma4Config)
    monkeypatch.setattr(tr, "nnx", MockNNX())

    def mock_raise_error(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(tr, "build_dataloader", Exception)
    res = run_dpo(DPOConfig(model_name="model", dataset="data"))
    if not res["backend"] == "jax":
        raise AssertionError


def test_run_dpo_jax_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(tr, "jax", "mock")
    monkeypatch.setattr(tr, "jnp", MockJnp())
    monkeypatch.setattr(tr, "jnn", MockJnn())
    monkeypatch.setattr(tr, "optax", MockOptax())
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(tr, "Gemma4Config", MockGemma4Config)
    monkeypatch.setattr(tr, "nnx", MockNNX())

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """

        class MockLoader:
            """Provide class docstring."""

            def __iter__(self) -> object:
                """Execute function.

                Yields:
                    object: Description of yield.

                """
                yield {"chosen_inputs": MockArray(), "chosen_labels": MockArray(), "rejected_inputs": MockArray(), "rejected_labels": MockArray()}

            def __len__(self) -> int:
                """Execute function.

                Returns:
                    object: Description of return.

                """
                return 1

        return {"loader": MockLoader()}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    res = run_dpo(DPOConfig(model_name="model", dataset="data"))
    if res["status"] != "completed":
        raise AssertionError


def test_run_dpo_jax_no_dataloader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(tr, "jax", "mock")
    monkeypatch.setattr(tr, "jnp", MockJnp())
    monkeypatch.setattr(tr, "jnn", MockJnn())
    monkeypatch.setattr(tr, "optax", MockOptax())
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(tr, "Gemma4Config", MockGemma4Config)
    monkeypatch.setattr(tr, "nnx", MockNNX())

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return {"loader": None}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    res = run_dpo(DPOConfig(model_name="model", dataset="data"))
    if res["status"] != "completed":
        raise AssertionError


def test_run_dpo_jax_error_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(tr, "jax", "mock")
    monkeypatch.setattr(tr, "jnp", MockJnp())
    monkeypatch.setattr(tr, "jnn", MockJnn())
    monkeypatch.setattr(tr, "optax", MockOptax())
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(tr, "Gemma4Config", MockGemma4Config)
    monkeypatch.setattr(tr, "nnx", MockNNX())

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> Never:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "simulated error"
        raise ValueError(msg)

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    res = run_dpo(DPOConfig(model_name="model", dataset="data"))
    if "failed: simulated error" not in res["status"]:
        raise AssertionError


def test_dpo_loss_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        TypeError: Description.

    """
    monkeypatch.setattr(tr, "jnp", MockJnp())
    monkeypatch.setattr(tr, "jnn", MockJnn())
    (a, _b, _c) = dpo_loss(MockArray(), MockArray(), MockArray(), MockArray())
    if not isinstance(a, MockArray):
        raise TypeError


def test_dpo_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(tr)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "flax", None)
    importlib.reload(tr)
    monkeypatch.undo()
    importlib.reload(tr)
