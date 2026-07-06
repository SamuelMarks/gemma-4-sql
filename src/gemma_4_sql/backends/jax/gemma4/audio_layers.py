"""Audio layers and submodules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

from .layers import Gemma4ClippableLinear, Gemma4RMSNorm

if TYPE_CHECKING:
    from .config import AudioConfig


class Gemma4AudioSubSampleConvProjectionLayer(nnx.Module):
    """A single convolutional projection layer for audio subsampling."""

    def __init__(self, in_channels: int, channels: int, norm_eps: float, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__.

        Args:
            in_channels: The integer value for in channels.
            channels: The integer value for channels.
            norm_eps: The float value for norm eps.
            rngs: The rngs.
        """
        self.conv = nnx.Conv(in_channels, channels, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), use_bias=False, rngs=rngs)
        self.norm = nnx.LayerNorm(channels, epsilon=norm_eps, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> tuple[jax.Array, jax.Array | None]:
        """Apply the subsample convolution projection layer.

        Returns:
            object: The resulting output from the operation.

        """
        if mask is not None:
            x *= mask[:, None, :, None]
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
        """Apply the full subsample convolution projection.

        Returns:
            object: The resulting output from the operation.

        """
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
        """Apply the feed forward network.

        Returns:
            object: The resulting output from the operation.

        """
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
        """Docstring for __init__.

        Args:
            x: The input x.

        Returns:
            The execution result.
        """
        self.kernel_size = config.conv_kernel_size
        self.left_pad = self.kernel_size - 1
        self.conv = nnx.Conv(config.hidden_size, config.hidden_size, kernel_size=self.kernel_size, feature_group_count=config.hidden_size, use_bias=False, padding=0, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply causal 1D convolution.

        Args:
            config: The configuration parameters.
            rngs: The rngs.
        """
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
        """Apply lightweight 1D convolution.

        Returns:
            object: The resulting output from the operation.

        """
        residual = x
        x = self.pre_layer_norm(x)
        x = self.linear_start(x)
        (x, gate) = jnp.split(x, 2, axis=-1)
        x *= jax.nn.sigmoid(gate)
        x = self.depthwise_conv1d(x)
        x = jnp.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.conv_norm(x)
        x = jax.nn.silu(x)
        x = self.linear_end(x)
        return residual + x
