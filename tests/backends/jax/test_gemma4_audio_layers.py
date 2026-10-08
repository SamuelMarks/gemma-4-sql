"""Module docstring."""

from unittest.mock import MagicMock

import jax.numpy as jnp
from flax import nnx

from gemma_4_sql.backends.jax.gemma4.audio_layers import (
    Gemma4AudioCausalConv1d,
    Gemma4AudioFeedForward,
    Gemma4AudioLightConv1d,
    Gemma4AudioSubSampleConvProjection,
    Gemma4AudioSubSampleConvProjectionLayer,
)


def get_mock_audio_config():
    """Docstring for get_mock_audio_config."""
    config = MagicMock()
    config.hidden_size = 16
    config.subsampling_conv_channels = (4, 4)
    config.rms_norm_eps = 1e-6
    config.use_clipped_linears = False
    config.gradient_clipping = 100.0
    config.residual_weight = 0.5
    config.conv_kernel_size = 3
    return config


def test_subsample_conv_projection_layer():
    """Docstring for test_subsample_conv_projection_layer."""
    rngs = nnx.Rngs(0)
    layer = Gemma4AudioSubSampleConvProjectionLayer(1, 4, 1e-6, rngs=rngs)

    x = jnp.ones((2, 1, 8, 16))  # batch, channel, seq_len, feat
    out, _mask = layer(x)
    assert out.shape == (2, 4, 4, 8)

    mask_in = jnp.ones((2, 8))
    _out2, mask2 = layer(x, mask_in)
    assert mask2.shape == (2, 4)


def test_subsample_conv_projection():
    """Docstring for test_subsample_conv_projection."""
    config = get_mock_audio_config()
    rngs = nnx.Rngs(0)
    proj = Gemma4AudioSubSampleConvProjection(config, rngs=rngs)

    x = jnp.ones((2, 8, 4))
    out, _mask = proj(x)
    assert out.shape == (2, 2, 16)  # seq_len 8 -> 4 -> 2

    mask_in = jnp.ones((2, 8))
    _out2, mask2 = proj(x, mask_in)
    assert mask2.shape == (2, 2)


def test_feed_forward():
    """Docstring for test_feed_forward."""
    config = get_mock_audio_config()
    rngs = nnx.Rngs(0)
    ffw = Gemma4AudioFeedForward(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = ffw(x)
    assert out.shape == (2, 4, 16)


def test_causal_conv1d():
    """Docstring for test_causal_conv1d."""
    config = get_mock_audio_config()
    rngs = nnx.Rngs(0)
    conv = Gemma4AudioCausalConv1d(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = conv(x)
    assert out.shape == (2, 4, 16)


def test_light_conv1d():
    """Docstring for test_light_conv1d."""
    config = get_mock_audio_config()
    rngs = nnx.Rngs(0)
    lconv = Gemma4AudioLightConv1d(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = lconv(x)
    assert out.shape == (2, 4, 16)
