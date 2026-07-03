"""Tests for JAX PEFT."""

import pytest

import gemma_4_sql.backends.jax.peft as pt


class MockOptax:
    """Mock optax."""


class MockJax:
    """Mock jax."""


class MockGemma4Config:
    """Provide class docstring."""

    @staticmethod
    def gemma4_e2b() -> object:
        """Execute function."""
        return "config"


class MockGemma4ForCausalLM:
    """Provide class docstring."""

    def __init__(self, config: object, rngs: object) -> None:
        """Execute function."""


class MockNNX:
    """Provide class docstring."""

    class Param:
        """Provide class docstring."""

    class Rngs:
        """Provide class docstring."""

        def __init__(self, seed: int) -> None:
            """Execute function."""

    @staticmethod
    def split(model: object, *_args: object, **_kwargs: object) -> tuple:
        """Execute function."""
        return (model, {}, {})


def test_apply_lora_jax_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(pt, "optax", None)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "mocked_missing_optax":
        raise AssertionError


def test_apply_lora_jax_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(pt, "optax", MockOptax())
    monkeypatch.setattr(pt, "jax", MockJax())
    monkeypatch.setattr(pt, "nnx", MockNNX())
    monkeypatch.setattr(pt, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(pt, "Gemma4Config", MockGemma4Config)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "completed":
        raise AssertionError


def test_apply_lora_jax_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(pt, "optax", MockOptax())
    monkeypatch.setattr(pt, "jax", MockJax())
    monkeypatch.setattr(pt, "nnx", MockNNX())
    monkeypatch.setattr(pt, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(pt, "Gemma4Config", MockGemma4Config)

    def mock_split(*_args: object, **_kwargs: object) -> tuple:
        """Execute function."""
        msg = "split error"
        raise ValueError(msg)

    monkeypatch.setattr(MockNNX, "split", mock_split)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if "failed" not in str(res["status"]):
        raise AssertionError


def test_peft_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib")
    sys = __import__("sys")
    mdl = __import__("gemma_4_sql.backends.jax.peft")
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "flax", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    importlib.reload(mdl)
