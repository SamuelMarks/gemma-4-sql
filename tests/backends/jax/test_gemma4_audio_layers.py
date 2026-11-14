"""Unit and behavioral tests for gemma4 audio layers."""

from __future__ import annotations

from tests.conftest import requires_jax

pytestmark = requires_jax

from flax import nnx

from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.jax.gemma4.audio_layers import (
    Gemma4AudioCausalConv1d,
    Gemma4AudioFeedForward,
    Gemma4AudioLightConv1d,
    Gemma4AudioSubSampleConvProjection,
    Gemma4AudioSubSampleConvProjectionLayer,
)
from gemma_4_sql.backends.jax.gemma4.config import AudioConfig


def make_audio_config() -> AudioConfig:
    """Create a miniature `AudioConfig` instance for audio layer tests.

    Returns:
        A small `AudioConfig` configured for audio convolution and FFN tests.

    """
    return AudioConfig(
        hidden_size=16,
        subsampling_conv_channels=(4, 4),
        rms_norm_eps=1e-6,
        use_clipped_linears=False,
        gradient_clipping=100.0,
        residual_weight=0.5,
        conv_kernel_size=3,
    )


def test_subsample_conv_projection_layer() -> None:
    """Verify subsample conv projection layer behavior."""
    rngs = nnx.Rngs(0)
    layer = Gemma4AudioSubSampleConvProjectionLayer(1, 4, 1e-6, rngs=rngs)

    x = jnp.ones((2, 1, 8, 16))
    out, _mask = layer(x)
    assert out.shape == (2, 4, 4, 8)

    mask_in = jnp.ones((2, 8))
    _out2, mask2 = layer(x, mask_in)
    assert mask2 is not None and mask2.shape == (2, 4)


def test_subsample_conv_projection() -> None:
    """Verify subsample conv projection behavior."""
    config = make_audio_config()
    rngs = nnx.Rngs(0)
    proj = Gemma4AudioSubSampleConvProjection(config, rngs=rngs)

    x = jnp.ones((2, 8, 4))
    out, _mask = proj(x)
    assert out.shape == (2, 2, 16)

    mask_in = jnp.ones((2, 8))
    _out2, mask2 = proj(x, mask_in)
    assert mask2 is not None and mask2.shape == (2, 2)


def test_feed_forward() -> None:
    """Verify feed forward behavior."""
    config = make_audio_config()
    rngs = nnx.Rngs(0)
    ffw = Gemma4AudioFeedForward(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = ffw(x)
    assert out.shape == (2, 4, 16)


def test_causal_conv1d() -> None:
    """Verify causal conv1d behavior."""
    config = make_audio_config()
    rngs = nnx.Rngs(0)
    conv = Gemma4AudioCausalConv1d(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = conv(x)
    assert out.shape == (2, 4, 16)


def test_light_conv1d() -> None:
    """Verify light conv1d behavior."""
    config = make_audio_config()
    rngs = nnx.Rngs(0)
    lconv = Gemma4AudioLightConv1d(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = lconv(x)
    assert out.shape == (2, 4, 16)
