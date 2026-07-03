"""Provide module docstring."""

from typing import Never

import jax
import jax.numpy as jnp
import pytest

from gemma_4_sql.backends.jax.gemma4.params import create_gemma4_from_pretrained


def test_assign_weight_type_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    m_params = __import__("gemma_4_sql.backends.jax.gemma4.params")

    def raise_type_error(*_args: object, **_kwargs: object) -> Never:
        """Execute function."""
        msg = "err"
        raise TypeError(msg)

    monkeypatch.setattr(m_params, "assign_weights_from_eval_shape", raise_type_error)
    with pytest.raises(TypeError):
        m_params.process_standard_tensor(type("SF", (), {"get_tensor": lambda _self, _k: 1})(), "a", {}, {"a": ("a", type("MockTransform", (), {"value": None})())})


class MockConfig:
    """Provide class docstring."""

    hidden_size = 128
    vision_config = True


class MockModelObj:
    """Provide class docstring."""

    vision_tower = type("VT", (), {"embeddings": type("E", (), {"num_patches": 10})()})()


class MockNNX:
    """Provide class docstring."""

    def eval_shape(self, _fn: object) -> object:
        """Execute function."""
        return MockModelObj()

    def split(self, _x: object) -> object:
        """Execute function."""
        return (None, type("State", (), {"to_pure_dict": lambda _self: {"model": {"embed_scale": jax.ShapeDtypeStruct((1,), jnp.bfloat16)}, "vision_tower": {"embeddings": {"position_ids": jax.ShapeDtypeStruct((1,), jnp.int32)}}}})())

    def merge(self, _graph_def: object, state: object) -> object:
        """Execute function."""
        return state

    class Rngs:
        """Provide class docstring."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Execute function."""


class MockSt:
    """Provide class docstring."""

    def safe_open(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return type("CM", (), {"__enter__": lambda _self: _self, "__exit__": lambda _self, *_a: None, "keys": lambda _self: [], "get_tensor": lambda _self, _k: jnp.array(1)})()


def test_create_gemma4_from_pretrained_missing_nnx_state(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function."""
    m_params = __import__("gemma_4_sql.backends.jax.gemma4.params")
    monkeypatch.setattr(m_params, "nnx", MockNNX())
    monkeypatch.setattr(m_params, "safetensors", MockSt())
    (tmp_path / "model.safetensors").touch()
    res = create_gemma4_from_pretrained(str(tmp_path), MockConfig())
    if "embed_scale" not in res["model"]:
        raise AssertionError
    if "position_ids" not in res["vision_tower"]["embeddings"]:
        raise AssertionError
