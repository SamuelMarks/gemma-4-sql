# Copyright 2024
"""Audio attention implementation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

from .layers import ConstVar, Gemma4ClippableLinear

if TYPE_CHECKING:
    from .config import AudioConfig


class Gemma4AudioRelPositionalEncoding(nnx.Module):
    """Sinusoidal relative positional encoding for the audio encoder."""

    def __init__(self, config: AudioConfig) -> None:
        """Docstring for __init__."""
        self.hidden_size = config.hidden_size
        self.context_size = config.attention_chunk_size + config.attention_context_left - 1 + config.attention_context_right
        min_timescale = 1.0
        max_timescale = 10000.0
        num_timescales = self.hidden_size // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales) * -log_timescale_increment)
        self.inv_timescales = ConstVar(inv_timescales[None, None, :])

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply relative positional encoding.

        Returns:
            object: The resulting output from the operation.

        """
        position_ids = jnp.arange(self.context_size // 2, -1, -1, dtype=x.dtype)
        position_ids = position_ids[..., None]
        scaled_time = position_ids * self.inv_timescales[...]
        pos_embed = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)
        return pos_embed.astype(x.dtype)


def _convert_to_block(x: jax.Array, chunk_size: int) -> jax.Array:
    """Reshapes the input into chunks/blocks for block-wise attention.

    Returns:
        object: The resulting output from the operation.

    """
    (batch_size, seq_len, num_heads, head_dim) = x.shape
    num_blocks = (seq_len + chunk_size - 1) // chunk_size
    pad_len = num_blocks * chunk_size - seq_len
    x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
    return x.reshape(batch_size, num_blocks, chunk_size, num_heads, head_dim)


def _extract_block_context(x: jax.Array, attn: object) -> jax.Array:
    """Extract the left context block for block-wise attention.

    Returns:
        object: The resulting output from the operation.

    """
    (batch_size, seq_len, num_heads, head_dim) = x.shape
    x = jnp.pad(x, ((0, 0), (attn.max_past_horizon, attn.max_future_horizon + attn.chunk_size - 1), (0, 0), (0, 0)))
    num_blocks = (seq_len + attn.chunk_size - 1) // attn.chunk_size
    blocks = []
    for i in range(num_blocks):
        start = i * attn.chunk_size
        blocks.append(jax.lax.dynamic_slice(x, (0, start, 0, 0), (batch_size, attn.context_size, num_heads, head_dim)))
    return jnp.stack(blocks, axis=1)


def _rel_shift(x: jax.Array, context_size: int) -> jax.Array:
    """Perform relative shift on attention scores.

    Returns:
        object: The resulting output from the operation.

    """
    (batch_size, num_heads, num_blocks, block_size, position_length) = x.shape
    x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 0), (0, context_size + 1 - position_length)))
    x = x.reshape((batch_size, num_heads, num_blocks, block_size * (context_size + 1)))
    x = x[..., : block_size * context_size]
    return x.reshape((batch_size, num_heads, num_blocks, block_size, context_size))


def _compute_audio_attention_outputs(attn: object, qkv: tuple[jax.Array, jax.Array, jax.Array], pos_emb: jax.Array, mask: jax.Array | None) -> jax.Array:
    """Compute the multi-head attention outputs for audio.

    Returns:
        object: The resulting output from the operation.

    """
    (q, k, v) = qkv
    (batch_size, seq_len, _) = q.shape[:3]
    q = q * attn.q_scale * jax.nn.softplus(attn.per_dim_scale[...])
    k *= attn.k_scale
    q_block = _convert_to_block(q, attn.chunk_size)
    k_context = _extract_block_context(k, attn)
    v_context = _extract_block_context(v, attn)
    num_blocks = q_block.shape[1]
    rel_k = attn.relative_k_proj(pos_emb).reshape((-1, attn.num_heads, attn.head_dim)).astype(q.dtype)
    queries = jnp.transpose(q_block, (0, 3, 1, 2, 4))
    keys = jnp.transpose(k_context, (0, 3, 1, 4, 2))
    matrix_ac = jnp.matmul(queries, keys)
    queries_flat = queries.reshape((batch_size, attn.num_heads, -1, attn.head_dim))
    rel_k_t = jnp.transpose(rel_k, (1, 2, 0))
    matrix_bd = jnp.matmul(queries_flat, rel_k_t)
    matrix_bd = matrix_bd.reshape((batch_size, attn.num_heads, num_blocks, attn.chunk_size, -1))
    matrix_bd = _rel_shift(matrix_bd, attn.context_size)
    attn_weights = matrix_ac + matrix_bd
    attn_weights /= attn.softcap
    attn_weights = jnp.tanh(attn_weights) * attn.softcap
    if mask is not None:
        attn_weights = jnp.where(mask, attn_weights, attn.invalid_logits_value)
    attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(v_context.dtype)
    values = jnp.transpose(v_context, (0, 3, 1, 2, 4))
    out = jnp.matmul(attn_weights, values)
    out = jnp.transpose(out, (0, 2, 3, 1, 4))
    out = out.reshape((batch_size, num_blocks * attn.chunk_size, -1))
    out = out[:, :seq_len, :]
    return attn.post(out)


class Gemma4AudioAttention(nnx.Module):
    """Chunked local attention with relative position bias for audio."""

    def __init__(self, config: AudioConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_scale = self.head_dim ** (-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)
        self.chunk_size = config.attention_chunk_size
        self.max_past_horizon = config.attention_context_left - 1
        self.max_future_horizon = config.attention_context_right
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon
        self.softcap = config.attention_logit_cap
        self.invalid_logits_value = config.attention_invalid_logits_value
        hs = config.hidden_size
        self.q_proj = Gemma4ClippableLinear(hs, self.num_heads * self.head_dim, use_clipped_linears=config.use_clipped_linears, rngs=rngs)
        self.k_proj = Gemma4ClippableLinear(hs, self.num_heads * self.head_dim, use_clipped_linears=config.use_clipped_linears, rngs=rngs)
        self.v_proj = Gemma4ClippableLinear(hs, self.num_heads * self.head_dim, use_clipped_linears=config.use_clipped_linears, rngs=rngs)
        self.post = Gemma4ClippableLinear(hs, hs, use_clipped_linears=config.use_clipped_linears, rngs=rngs)
        self.relative_k_proj = nnx.Linear(hs, self.num_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.per_dim_scale = nnx.Param(jnp.zeros(self.head_dim))

    def __call__(self, x: jax.Array, pos_emb: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """Compute the multi-head attention for audio inputs.

        Returns:
            object: The resulting output from the operation.

        """
        (batch_size, seq_len, _) = x.shape
        q = self.q_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        k = self.k_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        v = self.v_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        return _compute_audio_attention_outputs(self, (q, k, v), pos_emb, mask)
