"""Unit and behavioral tests for gemma4 audio."""

from __future__ import annotations

from tests.conftest import requires_jax

pytestmark = requires_jax

from flax import nnx

from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.jax.gemma4.audio import Gemma4AudioLayer, Gemma4AudioModel
from gemma_4_sql.backends.jax.gemma4.config import AudioConfig


def make_audio_config(*, use_clipped_linears: bool = False) -> AudioConfig:
    """Create a miniature `AudioConfig` instance for unit tests.

    Returns:
        A small `AudioConfig` for fast audio layer testing.

    """
    return AudioConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        hidden_act="silu",
        subsampling_conv_channels=(4, 4),
        conv_kernel_size=3,
        residual_weight=0.5,
        attention_chunk_size=2,
        attention_context_left=2,
        attention_context_right=1,
        attention_logit_cap=50.0,
        attention_invalid_logits_value=1e-9,
        use_clipped_linears=use_clipped_linears,
        gradient_clipping=100.0,
        output_proj_dims=8,
        rms_norm_eps=1e-6,
    )


def test_gemma4_audio_layer() -> None:
    """Verify gemma4 audio layer behavior."""
    config = make_audio_config()
    rngs = nnx.Rngs(0)
    layer = Gemma4AudioLayer(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    pos_emb = jnp.ones((1, 4, 16))

    out = layer(x, pos_emb)
    assert out.shape == (2, 4, 16)


def test_gemma4_audio_model() -> None:
    """Verify gemma4 audio model behavior."""
    config = make_audio_config()
    rngs = nnx.Rngs(0)
    model = Gemma4AudioModel(config, rngs=rngs)

    input_features = jnp.ones((2, 8, 4))

    out = model(input_features)
    assert out.shape == (2, 2, 8)

    mask = jnp.ones((2, 8))
    out_mask = model(input_features, attention_mask=mask)
    assert out_mask.shape == (2, 2, 8)


def test_gemma4_audio_layer_clipped() -> None:
    """Verify gemma4 audio layer clipped behavior."""
    config = make_audio_config(use_clipped_linears=True)
    rngs = nnx.Rngs(0)
    layer = Gemma4AudioLayer(config, rngs=rngs)
    x = jnp.ones((2, 4, 16))
    pos_emb = jnp.ones((1, 4, 16))
    out = layer(x, pos_emb)
    assert out.shape == (2, 4, 16)
