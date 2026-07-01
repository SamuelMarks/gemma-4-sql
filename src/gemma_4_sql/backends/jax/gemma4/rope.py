"""Module docstring."""

from __future__ import annotations

import typing
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

if TYPE_CHECKING:
    from jaxtyping import Array  # pragma: no cover


def segment_ids_to_positions(segment_ids: Array) -> Array:
    """Initialize function segment_ids_to_positions.

    Args:
    ----
    segment_ids: Description of segment_ids.

    """
    return jnp.cumsum(segment_ids, axis=-1)


def default_rope_params(_positions: Array, head_dim: int, rope_theta: int = 1000000, factor: float = 1.0) -> tuple[Array, Array]:
    """Initialize function default_rope_params.

    Args:
    ----
    positions: Description of positions.
    head_dim: Description of head_dim.
    rope_theta: Description of rope_theta.
    factor: Description of factor.

    """
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction
    rotational_frequency = 1.0 / timescale
    rotational_frequency /= factor
    attention_factor = 1.0
    return (rotational_frequency, attention_factor)


rope_functions = {"default": default_rope_params}


def apply_rope(x: Array, sin: Array, cos: Array) -> Array:
    """Initialize function apply_rope.

    Args:
    ----
    x: Description of x.
    sin: Description of sin.
    cos: Description of cos.

    """
    if not (x.ndim == int("4") and sin.ndim == int("3") and (cos.ndim == int("3"))):
        raise AssertionError
    (x1, x2) = (x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :])
    (sin, cos) = (sin[:, :, None, :], cos[:, :, None, :])
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


class RoPE(nnx.Module):  # type: ignore[misc]
    """Initialize class RoPE."""

    def __init__(self: typing.Any, *, rope_type: str, **rope_kwargs: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        rope_type: Description of rope_type.
        rope_kwargs: Description of rope_kwargs.

        """
        self.rope_kwargs = rope_kwargs
        self.rope_fn = partial(rope_functions[rope_type], **rope_kwargs)

    def __call__(self: typing.Any, positions: Array) -> tuple[Array, Array]:
        """Initialize function __call__.

        Args:
        ----
        positions: Description of positions.

        """
        (rotational_frequency, attention_factor) = self.rope_fn(positions)
        sinusoid_inp = jnp.einsum("BT,k->BTk", positions, rotational_frequency, precision=jax.lax.Precision.HIGHEST)
        (sin, cos) = (jnp.sin(sinusoid_inp) * attention_factor, jnp.cos(sinusoid_inp) * attention_factor)
        return (sin, cos)
