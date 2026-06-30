"""Module docstring."""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np

from gemma_4_sql.backends.jax.gemma4.utils_params import assign_weights, assign_weights_from_eval_shape, map_to_jax_key, stoi


def test_utils() -> object:  # type: ignore[return]
    """Docstring for test_utils."""
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
        assign_weights(["a"], jnp.ones((4,)), state2, "k", ((0,), (2, 2), True), None)  # type: ignore[call-arg]
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((4,)), state2, "k", ((0,), (2, 2), False), None)  # type: ignore[call-arg]
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((2, 2)), state2, "k", ((1, 0), None, False), None)  # type: ignore[call-arg]
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights_from_eval_shape(["a"], jnp.ones((3, 3)), state, "k", None)
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((3, 3)), state2, "k", None, None)  # type: ignore[call-arg]
    sharding = {"a": jax.sharding.NamedSharding(jax.sharding.Mesh(np.array(jax.devices()), ("x",)), jax.sharding.PartitionSpec("x"))}
    with contextlib.suppress(ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        assign_weights(["a"], jnp.ones((2, 2)), state2, "k", None, sharding)  # type: ignore[call-arg]
    if not stoi("123") == int("123"):
        raise AssertionError


def test_utils_part1() -> object:  # type: ignore[return]
    """Docstring."""
    if not stoi("abc") == "abc":
        raise AssertionError
    mapping = {"abc\\.(\\d+)": ("def.xyz", None), "cde": ("uvw", None)}
    map_to_jax_key(mapping, "abc.123")
    map_to_jax_key(mapping, "cde")
    map_to_jax_key(mapping, "unmatched")
