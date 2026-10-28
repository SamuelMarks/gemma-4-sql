"""Rotary Position Embedding (RoPE) implementation for Gemma 4 in JAX."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array


def segment_ids_to_positions(segment_ids: Array) -> Array:
    """Convert segment mask IDs into cumulative token position indices.

    Args:
        segment_ids: Segment indicator array.

    Returns:
        Cumulative position index array along the last axis.

    """
    return jnp.cumsum(segment_ids, axis=-1)


def default_rope_params(_positions: Array, head_dim: int, rope_theta: int = 1000000, factor: float = 1.0) -> tuple[Array, float]:
    """Compute default inverse rotational frequencies and scaling factor for RoPE.

    Args:
        _positions: Token position tensor (unused for static frequency computation).
        head_dim: Attention head dimension.
        rope_theta: Base period for rotary frequencies.
        factor: Proportional scaling factor for frequencies.

    Returns:
        Tuple of `(rotational_frequency, attention_factor)`.

    """
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = float(rope_theta) ** fraction
    rotational_frequency = 1.0 / timescale
    rotational_frequency /= factor
    attention_factor = 1.0
    return (rotational_frequency, attention_factor)


rope_functions: dict[str, Callable[..., tuple[Array, float]]] = {"default": default_rope_params}


def apply_rope(x: Array, sin: Array, cos: Array) -> Array:
    """Apply rotary position embeddings to query or key projections.

    Args:
        x: 4D query or key projection tensor `(batch, seq_len, heads, head_dim)`.
        sin: 3D sine frequency tensor `(batch, seq_len, head_dim // 2)`.
        cos: 3D cosine frequency tensor `(batch, seq_len, head_dim // 2)`.

    Returns:
        Rotated query or key tensor preserving `x.dtype`.

    Raises:
        ValueError: If input tensor ranks do not match `(4, 3, 3)`.

    """
    if not (x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3):
        msg = f"Expected (x.ndim, sin.ndim, cos.ndim) == (4, 3, 3), got ({x.ndim}, {sin.ndim}, {cos.ndim})"
        raise ValueError(msg)
    (x1, x2) = (x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :])
    (sin, cos) = (sin[:, :, None, :], cos[:, :, None, :])
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


class RoPE(nnx.Module):
    """Implementation of RoPE."""

    def __init__(self, *, rope_type: str, **rope_kwargs: float) -> None:
        """Initialize the instance parameters.

        Args:
            rope_type: The rope_type parameter required for this operation.
            **rope_kwargs: Additional configuration for RoPE embeddings.

        """
        self.rope_kwargs = rope_kwargs
        self.rope_fn = partial(rope_functions[rope_type], **rope_kwargs)

    def __call__(self, positions: Array) -> tuple[Array, Array]:
        """Compute sine and cosine rotary tables for given token positions.

        Args:
            positions: 2D token position indices `(batch, seq_len)`.

        Returns:
            Tuple of `(sin, cos)` arrays.

        """
        (rotational_frequency, attention_factor) = self.rope_fn(positions)
        sinusoid_inp = jnp.einsum("BT,k->BTk", positions, rotational_frequency, precision=jax.lax.Precision.HIGHEST)
        (sin, cos) = (jnp.sin(sinusoid_inp) * attention_factor, jnp.cos(sinusoid_inp) * attention_factor)
        return (sin, cos)
