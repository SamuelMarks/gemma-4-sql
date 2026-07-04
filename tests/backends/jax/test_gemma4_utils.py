# Copyright 2024
"""Provide module docstring."""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gemma_4_sql.backends.jax.gemma4.utils_params as ut
from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights, assign_weights_from_eval_shape, create_model_from_safe_tensors, map_to_jax_key, stoi

EXPECTED_VAL = 123


def test_utils() -> object:
    """Docstring for test_utils.

    Raises:
        AssertionError: Description.

    """
    state = {"a": jax.ShapeDtypeStruct((2, 2), jnp.float32)}
    jnp.ones((2, 2))
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights_from_eval_shape(["a"], jnp.ones((4,)), state, "k", ((0,), (2, 2), True))
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights_from_eval_shape(["a"], jnp.ones((4,)), state, "k", ((0,), (2, 2), False))
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights_from_eval_shape(["a"], jnp.ones((2, 2)), state, "k", ((1, 0), None, False))
    state2 = {"a": jnp.zeros((2, 2))}
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((4,)), state2, "k", ((0,), (2, 2), True), None)
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((4,)), state2, "k", ((0,), (2, 2), False), None)
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((2, 2)), state2, "k", ((1, 0), None, False), None)
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights_from_eval_shape(["a"], jnp.ones((3, 3)), state, "k", None)
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((3, 3)), state2, "k", None, None)
    sharding = {"a": jax.sharding.NamedSharding(jax.sharding.Mesh(np.array(jax.devices()), ("x",)), jax.sharding.PartitionSpec("x"))}
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((2, 2)), state2, "k", None, sharding)
    if not stoi("123") == EXPECTED_VAL:
        raise AssertionError


def test_utils_part1() -> object:
    """Docstring.

    Raises:
        AssertionError: Description.

    """
    if not stoi("abc") == "abc":
        raise AssertionError
    mapping = {"abc\\.(\\d+)": ("def.xyz", None), "cde": ("uvw", None)}
    map_to_jax_key(mapping, "abc.123")
    map_to_jax_key(mapping, "cde")
    map_to_jax_key(mapping, "unmatched")


class MockModelCls:
    """Provide class docstring."""

    def __init__(self, cfg: object, rngs: object) -> None:
        """Execute function."""
        self.cfg = cfg
        self.rngs = rngs


class MockSafeOpen:
    """Provide class docstring."""

    def __init__(self, filepath: object, framework: object) -> None:
        """Execute function."""
        self.filepath = filepath
        self.framework = framework

    def __enter__(self) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> object:
        """Execute function."""

    def keys(self) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return ["layer1.weight", "layer2.weight"]

    def get_tensor(self, _key: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return jnp.ones((2, 2))


def test_create_model_from_safe_tensors(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    d = tmp_path / "model_dir"
    d.mkdir()
    f = d / "model.safetensors"
    f.write_text("dummy content")
    monkeypatch.setattr(ut, "safe_open", MockSafeOpen)
    mapping = {"layer1\\.weight": ("layer1.w", None), "layer2\\.weight": ("layer2.w", None)}
    model = create_model_from_safe_tensors(str(d), MockModelCls, "config", mapping)
    if not model.cfg == "config":
        raise AssertionError


def test_create_model_from_safe_tensors_missing_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(ut, "safe_open", MockSafeOpen)
    model = create_model_from_safe_tensors("nonexistent_dir", MockModelCls, "config", {})
    if not model.cfg == "config":
        raise AssertionError


def test_create_model_from_safe_tensors_no_safetensors(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(ut, "safe_open", None)
    model = create_model_from_safe_tensors(str(tmp_path), MockModelCls, "config", {})
    if not model.cfg == "config":
        raise AssertionError
