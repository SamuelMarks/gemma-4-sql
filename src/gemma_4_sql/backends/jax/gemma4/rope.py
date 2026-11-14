"""Rotary Position Embedding (RoPE) implementation for Gemma 4 in JAX."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
from flax import nnx
from jax import Array

from gemma_4_sql.backends.jax.gemma4._jnp import jnp


def segment_ids_to_positions(segment_ids: Array) -> Array:
    """Convert segment mask IDs into cumulative token position indices.

    Args:
        segment_ids: Segment indicator array.

    Returns:
        Cumulative position index array along the last axis.

    """
    return jnp.cumsum(segment_ids, axis=-1)


def default_rope_params(
    _positions: Array,
    head_dim: int,
    rope_theta: int = 1000000,
    factor: float = 1.0,
    proportion: float = 1.0,
) -> tuple[Array, float]:
    """Compute inverse rotational frequencies with optional linear scaling and proportional partial rotation.

    Args:
        _positions: Token position tensor (unused for static frequency computation).
        head_dim: Attention head dimension.
        rope_theta: Base period for rotary frequencies.
        factor: Linear context scaling divisor applied to rotary frequencies.
        proportion: Fraction `(0.0, 1.0]` of `head_dim` dimensions receiving rotary rotation;
            remaining dimensions receive zero frequency (`cos=1, sin=0`, pass-through).

    Returns:
        Tuple of `(rotational_frequency, attention_factor)`.

    """
    half_dim = head_dim // 2
    rotary_half_dim = max(0, min(half_dim, round(proportion * half_dim)))
    if rotary_half_dim == 0:
        return (jnp.zeros((half_dim,), dtype=jnp.float32), 1.0)
    rotary_dim = rotary_half_dim * 2
    fraction = jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / float(rotary_dim)
    timescale = float(rope_theta) ** fraction
    rotational_frequency = (1.0 / timescale) / float(factor)
    if rotary_half_dim < half_dim:
        pad_zeros = jnp.zeros((half_dim - rotary_half_dim,), dtype=jnp.float32)
        rotational_frequency = jnp.concatenate([rotational_frequency, pad_zeros], axis=0)
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
    """Rotary Position Embedding module supporting linear scaling and proportional partial rotation."""

    def __init__(self, *, rope_type: str, **rope_kwargs: float) -> None:
        """Initialize the RoPE module.

        Args:
            rope_type: Registered key in `rope_functions` (e.g., `'default'`).
            **rope_kwargs: Static parameters forwarded to the registered frequency builder.

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
