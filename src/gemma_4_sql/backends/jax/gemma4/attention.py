# Copyright 2024
"""Gemma 4 Attention implementation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .layers import Gemma4RMSNorm, _make_linear
from .rope import RoPE, apply_rope

MASK_PENALTY = -10000.0

if TYPE_CHECKING:
    from .cache import LayerCache
    from .config import AttentionType, ModelConfig


def _compute_attention_scores_and_output(qkv: tuple[jax.Array, jax.Array, jax.Array], attention_mask: jax.Array | None, config_params: tuple[int, float | None, int, int]) -> jax.Array:
    """Compute attention scores and output.

    Returns:
        object: The resulting output from the operation.

    """
    (q, k, v) = qkv
    (head_dim, soft_cap, num_kv_heads, num_heads) = config_params
    scale = 1.0 / math.sqrt(head_dim)
    if num_kv_heads != num_heads:  # pragma: no cover
        num_rep = num_heads // num_kv_heads
        k = jnp.repeat(k, num_rep, axis=2)
        v = jnp.repeat(v, num_rep, axis=2)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if soft_cap is not None:  # pragma: no cover
        scores /= soft_cap
        scores = jnp.tanh(scores) * soft_cap
    if attention_mask is not None:  # pragma: no cover
        scores += attention_mask
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return out.reshape((out.shape[0], out.shape[1], -1))


def _prepare_qkv_for_attention(qkv: tuple[jax.Array, jax.Array, jax.Array], positions: Array, rope: object, cache: LayerCache | None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Prepare QKV and return masks.

    Returns:
        object: The resulting output from the operation.

    """
    (q, k, v) = qkv
    (_batch_size, seq_len, _, _) = q.shape
    (sin, cos) = rope(positions)
    q = apply_rope(q, sin, cos)
    k = apply_rope(k, sin, cos)
    if cache is not None:
        slice_indices = (0, cache.cur_ind.value, 0, 0)
        cache.k_cache.value = jax.lax.dynamic_update_slice(cache.k_cache[...], k, slice_indices)
        cache.v_cache.value = jax.lax.dynamic_update_slice(cache.v_cache[...], v, slice_indices)
        k = cache.k_cache[...]
        v = cache.v_cache[...]
        mask_len = k.shape[1]
    else:
        mask_len = seq_len
    mask = jnp.arange(mask_len)[None, :] <= positions[:, :, None]
    window = positions[:, :, None] - jnp.arange(mask_len)[None, :]
    return (q, k, v, mask, window)


class Gemma4Attention(nnx.Module):
    """Multi-Head / Grouped-Query Attention for Gemma 4."""

    def _setup_heads(self, config: ModelConfig, attention_type: AttentionType) -> None:
        self.num_heads = config.num_attention_heads
        if getattr(attention_type, "name", str(attention_type).upper()) == "GLOBAL":
            self.num_kv_heads = config.num_global_key_value_heads if config.num_global_key_value_heads is not None else config.num_key_value_heads
            self.head_dim = config.global_head_dim if config.global_head_dim is not None else config.head_dim
            self.share_kv = config.share_kv_projections
        else:
            self.num_kv_heads = config.num_key_value_heads
            self.head_dim = config.head_dim
            self.share_kv = False

    def _setup_rope(self, config: ModelConfig, attention_type: AttentionType) -> None:
        if getattr(attention_type, "name", str(attention_type).upper()) == "GLOBAL":
            rope_factor = config.global_rope_proportion
            rope_theta = config.global_rope_max_timescale if config.global_rope_max_timescale is not None else config.rope_max_timescale
        else:
            rope_factor = config.local_rope_proportion
            rope_theta = config.local_rope_max_timescale if config.local_rope_max_timescale is not None else config.rope_max_timescale
        self.rope = RoPE(rope_type="default", head_dim=self.head_dim, rope_theta=rope_theta, factor=rope_factor)

    def __init__(self, config: ModelConfig, attention_type: AttentionType, *, rngs: nnx.Rngs) -> None:
        """Initialize Attention."""
        self.config = config
        self.attention_type = attention_type
        self._setup_heads(config, attention_type)
        self.hidden_size = config.hidden_size
        self.dtype = config.dtype
        shd = config.shd_cfg
        self.q_proj = _make_linear(self.hidden_size, self.num_heads * self.head_dim, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
        self.k_proj = _make_linear(self.hidden_size, self.num_kv_heads * self.head_dim, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
        if not self.share_kv:
            self.v_proj = _make_linear(self.hidden_size, self.num_kv_heads * self.head_dim, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
        else:
            self.v_proj = None
        self.o_proj = _make_linear(self.num_heads * self.head_dim, self.hidden_size, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        self._setup_rope(config, attention_type)

    @jax.named_scope("gemma4_attention")
    def __call__(self, x: Array, positions: Array, cache: LayerCache | None = None, attention_mask: Array | None = None) -> Array:
        """Apply attention over the input sequences.

        Returns:
            object: The resulting output from the operation.

        """
        (batch_size, seq_len, _) = x.shape
        q = self.q_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        k = self.k_proj(x).reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = k if self.share_kv else self.v_proj(x).reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        (q, k, v, mask, window) = _prepare_qkv_for_attention((q, k, v), positions, self.rope, cache)
        if getattr(self.attention_type, "name", str(self.attention_type).upper()) == "LOCAL_SLIDING":  # pragma: no cover
            mask &= window < self.config.sliding_window_size
        structural_mask = jnp.where(mask, 0.0, MASK_PENALTY).astype(q.dtype)[:, None, :, :]
        attention_mask = structural_mask if attention_mask is None else attention_mask + structural_mask
        out = _compute_attention_scores_and_output((q, k, v), attention_mask, (self.head_dim, self.config.attn_logits_soft_cap, self.num_kv_heads, self.num_heads))
        if cache is not None:
            cache.cur_ind.value += seq_len
        return self.o_proj(out).astype(self.dtype)
