"""Unit and behavioral tests for gemma4 audio."""

from unittest.mock import MagicMock

from tests.conftest import requires_jax

pytestmark = requires_jax

import jax.numpy as jnp
from flax import nnx

from gemma_4_sql.backends.jax.gemma4.audio import Gemma4AudioLayer, Gemma4AudioModel


def get_mock_audio_config() -> MagicMock:
    """Provide get mock audio config test helper."""
    config = MagicMock()
    config.hidden_size = 16
    config.num_hidden_layers = 1
    config.num_attention_heads = 2
    config.hidden_act = "silu"
    config.subsampling_conv_channels = (4, 4)
    config.conv_kernel_size = 3
    config.residual_weight = 0.5
    config.attention_chunk_size = 2
    config.attention_context_left = 2
    config.attention_context_right = 1
    config.attention_logit_cap = 50.0
    config.attention_invalid_logits_value = 1e-9
    config.use_clipped_linears = False
    config.gradient_clipping = 100.0
    config.output_proj_dims = 8
    config.rms_norm_eps = 1e-6
    return config


def test_gemma4_audio_layer() -> None:
    """Verify gemma4 audio layer behavior."""
    config = get_mock_audio_config()
    rngs = nnx.Rngs(0)
    layer = Gemma4AudioLayer(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    pos_emb = jnp.ones((1, 4, 16))

    out = layer(x, pos_emb)
    assert out.shape == (2, 4, 16)


def test_gemma4_audio_model() -> None:
    """Verify gemma4 audio model behavior."""
    config = get_mock_audio_config()
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
    config = get_mock_audio_config()
    config.use_clipped_linears = True
    rngs = nnx.Rngs(0)
    layer = Gemma4AudioLayer(config, rngs=rngs)
    x = jnp.ones((2, 4, 16))
    pos_emb = jnp.ones((1, 4, 16))
    out = layer(x, pos_emb)
    assert out.shape == (2, 4, 16)
