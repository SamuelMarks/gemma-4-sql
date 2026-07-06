"""Core functionality for the audio module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

from .audio_attention import (
    Gemma4AudioAttention,
    Gemma4AudioRelPositionalEncoding,
    _compute_audio_attention_outputs,
    _convert_to_block,
    _extract_block_context,
    _rel_shift,
)
from .audio_layers import (
    Gemma4AudioCausalConv1d,
    Gemma4AudioFeedForward,
    Gemma4AudioLightConv1d,
    Gemma4AudioSubSampleConvProjection,
    Gemma4AudioSubSampleConvProjectionLayer,
)
from .layers import Gemma4RMSNorm

__all__ = [
    "Gemma4AudioAttention",
    "Gemma4AudioCausalConv1d",
    "Gemma4AudioFeedForward",
    "Gemma4AudioLayer",
    "Gemma4AudioLightConv1d",
    "Gemma4AudioModel",
    "Gemma4AudioRelPositionalEncoding",
    "Gemma4AudioSubSampleConvProjection",
    "Gemma4AudioSubSampleConvProjectionLayer",
    "_compute_audio_attention_outputs",
    "_convert_to_block",
    "_extract_block_context",
    "_rel_shift",
]


if TYPE_CHECKING:
    from .config import AudioConfig


class Gemma4AudioLayer(nnx.Module):
    """A single layer of the audio transformer model."""

    def __init__(self, config: AudioConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__.

        Args:
            config: The configuration parameters.
            rngs: The rngs.
        """
        self.feed_forward1 = Gemma4AudioFeedForward(config, rngs=rngs)
        self.feed_forward2 = Gemma4AudioFeedForward(config, rngs=rngs)
        self.self_attn = Gemma4AudioAttention(config, rngs=rngs)
        self.lconv1d = Gemma4AudioLightConv1d(config, rngs=rngs)
        self.norm_pre_attn = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.norm_post_attn = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.norm_out = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.gradient_clipping = config.gradient_clipping

    def __call__(self, x: jax.Array, pos_emb: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """Apply a single audio transformer layer.

        Returns:
            object: The resulting output from the operation.

        """
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
        """Convert a 4D attention mask to a 5D blocked format.

        Returns:
            object: The resulting output from the operation.

        """
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
        """Forward pass for the Gemma 4 Audio model.

        Returns:
            object: The resulting output from the operation.

        """
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
