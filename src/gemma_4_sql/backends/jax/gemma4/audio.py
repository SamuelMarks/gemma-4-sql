"""Provide module docstring."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

from .layers import ConstVar, Gemma4ClippableLinear, Gemma4RMSNorm

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
        """Apply relative positional encoding."""
        position_ids = jnp.arange(self.context_size // 2, -1, -1, dtype=x.dtype)
        position_ids = position_ids[..., None]
        scaled_time = position_ids * self.inv_timescales[...]
        pos_embed = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)
        return pos_embed.astype(x.dtype)


def _convert_to_block(x: jax.Array, chunk_size: int) -> jax.Array:
    """Reshapes the input into chunks/blocks for block-wise attention."""
    (batch_size, seq_len, num_heads, head_dim) = x.shape
    num_blocks = (seq_len + chunk_size - 1) // chunk_size
    pad_len = num_blocks * chunk_size - seq_len
    x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
    return x.reshape(batch_size, num_blocks, chunk_size, num_heads, head_dim)


def _extract_block_context(x: jax.Array, chunk_size: int, max_past: int, max_future: int, context_size: int) -> jax.Array:
    """Extract the left context block for block-wise attention."""
    (batch_size, seq_len, num_heads, head_dim) = x.shape
    x = jnp.pad(x, ((0, 0), (max_past, max_future + chunk_size - 1), (0, 0), (0, 0)))
    num_blocks = (seq_len + chunk_size - 1) // chunk_size
    blocks = []
    for i in range(num_blocks):
        start = i * chunk_size
        blocks.append(jax.lax.dynamic_slice(x, (0, start, 0, 0), (batch_size, context_size, num_heads, head_dim)))
    return jnp.stack(blocks, axis=1)


def _rel_shift(x: jax.Array, context_size: int) -> jax.Array:
    """Perform relative shift on attention scores."""
    (batch_size, num_heads, num_blocks, block_size, position_length) = x.shape
    x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 0), (0, context_size + 1 - position_length)))
    x = x.reshape((batch_size, num_heads, num_blocks, block_size * (context_size + 1)))
    x = x[..., : block_size * context_size]
    return x.reshape((batch_size, num_heads, num_blocks, block_size, context_size))


def _compute_audio_attention_outputs(attn: object, qkv: tuple[jax.Array, jax.Array, jax.Array], pos_emb: jax.Array, mask: jax.Array | None) -> jax.Array:
    """Compute the multi-head attention outputs for audio."""
    (q, k, v) = qkv
    (batch_size, seq_len, _) = q.shape[:3]
    q = q * attn.q_scale * jax.nn.softplus(attn.per_dim_scale[...])
    k = k * attn.k_scale
    q_block = _convert_to_block(q, attn.chunk_size)
    k_context = _extract_block_context(k, attn.chunk_size, attn.max_past_horizon, attn.max_future_horizon, attn.context_size)
    v_context = _extract_block_context(v, attn.chunk_size, attn.max_past_horizon, attn.max_future_horizon, attn.context_size)
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
    attn_weights = attn_weights / attn.softcap
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
        """Compute the multi-head attention for audio inputs."""
        (batch_size, seq_len, _) = x.shape
        q = self.q_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        k = self.k_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        v = self.v_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        return _compute_audio_attention_outputs(self, (q, k, v), pos_emb, mask)


class Gemma4AudioSubSampleConvProjectionLayer(nnx.Module):
    """A single convolutional projection layer for audio subsampling."""

    def __init__(self, in_channels: int, channels: int, norm_eps: float, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.conv = nnx.Conv(in_channels, channels, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), use_bias=False, rngs=rngs)
        self.norm = nnx.LayerNorm(channels, epsilon=norm_eps, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> tuple[jax.Array, jax.Array | None]:
        """Apply the subsample convolution projection layer."""
        if mask is not None:
            x = x * mask[:, None, :, None]
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = self.conv(x)
        x = self.norm(x)
        x = jax.nn.relu(x)
        x = jnp.transpose(x, (0, 3, 1, 2))
        if mask is not None:
            mask = mask[:, ::2]
        return (x, mask)


class Gemma4AudioSubSampleConvProjection(nnx.Module):
    """Full convolutional projection module for audio subsampling."""

    def __init__(self, config: AudioConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        (c0, c1) = config.subsampling_conv_channels
        self.layer0 = Gemma4AudioSubSampleConvProjectionLayer(1, c0, config.rms_norm_eps, rngs=rngs)
        self.layer1 = Gemma4AudioSubSampleConvProjectionLayer(c0, c1, config.rms_norm_eps, rngs=rngs)
        proj_input_dim = c0 // 4 * c1
        self.input_proj_linear = nnx.Linear(proj_input_dim, config.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> tuple[jax.Array, jax.Array | None]:
        """Apply the full subsample convolution projection."""
        x = jnp.expand_dims(x, 1)
        (x, mask) = self.layer0(x, mask)
        (x, mask) = self.layer1(x, mask)
        (batch_size, _, seq_len, _) = x.shape
        x = jnp.transpose(x, (0, 2, 3, 1)).reshape((batch_size, seq_len, -1))
        return (self.input_proj_linear(x), mask)


class Gemma4AudioFeedForward(nnx.Module):
    """Feed forward network used in the audio tower."""

    def __init__(self, config: AudioConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.ffw_layer_1 = Gemma4ClippableLinear(config.hidden_size, config.hidden_size * 4, use_clipped_linears=config.use_clipped_linears, rngs=rngs)
        self.ffw_layer_2 = Gemma4ClippableLinear(config.hidden_size * 4, config.hidden_size, use_clipped_linears=config.use_clipped_linears, rngs=rngs)
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.post_layer_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.gradient_clipping = config.gradient_clipping
        self.post_layer_scale = config.residual_weight

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the feed forward network."""
        residual = x
        x = jnp.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.pre_layer_norm(x)
        x = self.ffw_layer_1(x)
        x = jax.nn.silu(x)
        x = self.ffw_layer_2(x)
        x = jnp.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.post_layer_norm(x)
        x *= self.post_layer_scale
        return residual + x


class Gemma4AudioCausalConv1d(nnx.Module):
    """Causal 1D convolution layer for audio processing."""

    def __init__(self, config: AudioConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.kernel_size = config.conv_kernel_size
        self.left_pad = self.kernel_size - 1
        self.conv = nnx.Conv(config.hidden_size, config.hidden_size, kernel_size=self.kernel_size, feature_group_count=config.hidden_size, use_bias=False, padding=0, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply causal 1D convolution."""
        x = jnp.pad(x, ((0, 0), (self.left_pad, 0), (0, 0)))
        return self.conv(x)


class Gemma4AudioLightConv1d(nnx.Module):
    """Lightweight 1D convolution module for audio."""

    def __init__(self, config: AudioConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.linear_start = Gemma4ClippableLinear(config.hidden_size, config.hidden_size * 2, use_clipped_linears=config.use_clipped_linears, rngs=rngs)
        self.linear_end = Gemma4ClippableLinear(config.hidden_size, config.hidden_size, use_clipped_linears=config.use_clipped_linears, rngs=rngs)
        self.depthwise_conv1d = Gemma4AudioCausalConv1d(config, rngs=rngs)
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.conv_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.gradient_clipping = config.gradient_clipping

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply lightweight 1D convolution."""
        residual = x
        x = self.pre_layer_norm(x)
        x = self.linear_start(x)
        (x, gate) = jnp.split(x, 2, axis=-1)
        x = x * jax.nn.sigmoid(gate)
        x = self.depthwise_conv1d(x)
        x = jnp.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.conv_norm(x)
        x = jax.nn.silu(x)
        x = self.linear_end(x)
        return residual + x


class Gemma4AudioLayer(nnx.Module):
    """A single layer of the audio transformer model."""

    def __init__(self, config: AudioConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.feed_forward1 = Gemma4AudioFeedForward(config, rngs=rngs)
        self.feed_forward2 = Gemma4AudioFeedForward(config, rngs=rngs)
        self.self_attn = Gemma4AudioAttention(config, rngs=rngs)
        self.lconv1d = Gemma4AudioLightConv1d(config, rngs=rngs)
        self.norm_pre_attn = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.norm_post_attn = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.norm_out = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.gradient_clipping = config.gradient_clipping

    def __call__(self, x: jax.Array, pos_emb: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """Apply a single audio transformer layer."""
        x = self.feed_forward1(x)
        residual = x
        x = jnp.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.norm_pre_attn(x)
        x = self.self_attn(x, pos_emb, mask)
        x = jnp.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.norm_post_attn(x)
        x += residual
        x = self.lconv1d(x)
        x = self.feed_forward2(x)
        x = jnp.clip(x, -self.gradient_clipping, self.gradient_clipping)
        return self.norm_out(x)


class Gemma4AudioModel(nnx.Module):
    """An audio encoder based on the Universal Speech Model architecture."""

    def __init__(self, config: AudioConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.config = config
        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(config, rngs=rngs)
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(config)
        self.layers = nnx.List([Gemma4AudioLayer(config, rngs=rngs) for _ in range(config.num_hidden_layers)])
        self.output_proj = nnx.Linear(config.hidden_size, config.output_proj_dims, rngs=rngs)

    def _convert_4d_mask_to_blocked_5d(self, mask_4d: jax.Array) -> jax.Array:
        """Convert a 4D attention mask to a 5D blocked format."""
        (batch_size, _, seq_len, _) = mask_4d.shape
        chunk_size = self.config.attention_chunk_size
        max_past_horizon = self.config.attention_context_left - 1
        max_future_horizon = self.config.attention_context_right
        num_blocks = (seq_len + chunk_size - 1) // chunk_size
        padded_seq_len = num_blocks * chunk_size
        pad_amount = padded_seq_len - seq_len
        mask_4d = jnp.pad(mask_4d, ((0, 0), (0, pad_amount), (0, 0), (0, pad_amount)))
        mask_5d = mask_4d.reshape(batch_size, 1, num_blocks, chunk_size, padded_seq_len)
        mask_5d = jnp.pad(mask_5d, ((0, 0), (0, 0), (0, 0), (0, 0), (max_past_horizon, max_future_horizon)))
        block_starts = jnp.arange(num_blocks) * chunk_size
        offsets = jnp.arange(chunk_size + max_past_horizon + max_future_horizon)
        kv_indices = block_starts[:, None] + offsets[None, :]
        kv_indices = jnp.broadcast_to(kv_indices[None, None, :, None, :], (batch_size, 1, num_blocks, chunk_size, chunk_size + max_past_horizon + max_future_horizon))
        return jnp.take_along_axis(mask_5d, kv_indices, axis=-1)

    def __call__(self, input_features: jax.Array, attention_mask: jax.Array | None = None) -> jax.Array:
        """Forward pass for the Gemma 4 Audio model."""
        (x, mask) = self.subsample_conv_projection(input_features, attention_mask)
        pos_emb = self.rel_pos_enc(x)
        if mask is not None:
            mask_4d = mask[:, None, :, None] * mask[:, None, None, :]
            mask_5d = self._convert_4d_mask_to_blocked_5d(mask_4d)
        else:
            mask_5d = None
        for layer in self.layers:
            x = layer(x, pos_emb, mask_5d)
        return self.output_proj(x)
