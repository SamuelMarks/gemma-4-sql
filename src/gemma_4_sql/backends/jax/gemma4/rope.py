"""Core functionality for the rope module."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

if TYPE_CHECKING:
    from jaxtyping import Array


def segment_ids_to_positions(segment_ids: Array) -> Array:
    """Execute the segment ids to positions operation.

    Args:
        segment_ids: The segment ids.

    Returns:
        The resulting tensor array.
    """
    return jnp.cumsum(segment_ids, axis=-1)


def default_rope_params(_positions: Array, head_dim: int, rope_theta: int = 1000000, factor: float = 1.0) -> tuple[Array, Array]:
    """Execute the default rope params operation.

    Args:
    ----
    head_dim: The head_dim parameter required for this operation.
    rope_theta: The rope_theta parameter required for this operation.
    factor: The factor parameter required for this operation.


    Returns:
        object: The resulting output from the operation.

    """
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction
    rotational_frequency = 1.0 / timescale
    rotational_frequency /= factor
    attention_factor = 1.0
    return (rotational_frequency, attention_factor)


rope_functions = {"default": default_rope_params}


def apply_rope(x: Array, sin: Array, cos: Array) -> Array:
    """Execute the apply rope operation.

    Args:
    ----
    x: The x parameter required for this operation.
    sin: The sin parameter required for this operation.
    cos: The cos parameter required for this operation.


    Returns:
        object: The resulting output from the operation.

    Raises:
        AssertionError: If the test validation fails.

    """
    if not (x.ndim == int("4") and sin.ndim == int("3") and (cos.ndim == int("3"))):
        raise AssertionError
    (x1, x2) = (x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :])
    (sin, cos) = (sin[:, :, None, :], cos[:, :, None, :])
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


class RoPE(nnx.Module):
    """Implementation of RoPE."""

    def __init__(self, *, rope_type: str, **rope_kwargs: object) -> None:
        """Initialize the instance parameters.

        Args:
        ----
        rope_type: The rope_type parameter required for this operation.
        rope_kwargs: The rope_kwargs parameter required for this operation.

        """
        self.rope_kwargs = rope_kwargs
        self.rope_fn = partial(rope_functions[rope_type], **rope_kwargs)

    def __call__(self, positions: Array) -> tuple[Array, Array]:
        """Execute the callable logic.

        Args:
        ----
        positions: The positions parameter required for this operation.


        Returns:
            object: The resulting output from the operation.

        """
        (rotational_frequency, attention_factor) = self.rope_fn(positions)
        sinusoid_inp = jnp.einsum("BT,k->BTk", positions, rotational_frequency, precision=jax.lax.Precision.HIGHEST)
        (sin, cos) = (jnp.sin(sinusoid_inp) * attention_factor, jnp.cos(sinusoid_inp) * attention_factor)
        return (sin, cos)
