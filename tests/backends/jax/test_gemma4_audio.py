from unittest.mock import MagicMock

import jax.numpy as jnp
from flax import nnx

from gemma_4_sql.backends.jax.gemma4.audio import Gemma4AudioLayer, Gemma4AudioModel


def get_mock_audio_config():
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


def test_gemma4_audio_layer():
    config = get_mock_audio_config()
    rngs = nnx.Rngs(0)
    layer = Gemma4AudioLayer(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    pos_emb = jnp.ones((1, 4, 16))

    out = layer(x, pos_emb)
    assert out.shape == (2, 4, 16)


def test_gemma4_audio_model():
    config = get_mock_audio_config()
    rngs = nnx.Rngs(0)
    model = Gemma4AudioModel(config, rngs=rngs)

    # Subsampling expects input like (batch, channel, seq_len, feat)
    # Actually wait: subsampling does `jnp.expand_dims(x, 1)` -> `(batch, 1, seq_len, feat)`
    # Wait, the shape for x is typically (batch_size, seq_len, feat_dim)
    input_features = jnp.ones((2, 8, 4))

    out = model(input_features)
    assert out.shape == (2, 2, 8)  # subsampled seq_len will be 8/4 = 2, output_proj_dims=8

    # with mask
    mask = jnp.ones((2, 8))
    out_mask = model(input_features, attention_mask=mask)
    assert out_mask.shape == (2, 2, 8)


def test_gemma4_audio_layer_clipped():
    config = get_mock_audio_config()
    config.use_clipped_linears = True
    from flax import nnx

    rngs = nnx.Rngs(0)
    import jax.numpy as jnp

    from gemma_4_sql.backends.jax.gemma4.audio import Gemma4AudioLayer

    layer = Gemma4AudioLayer(config, rngs=rngs)
    x = jnp.ones((2, 4, 16))
    pos_emb = jnp.ones((1, 4, 16))
    out = layer(x, pos_emb)
    assert out.shape == (2, 4, 16)


def test_gemma4_audio_model_with_mask():
    config = get_mock_audio_config()
    from flax import nnx

    rngs = nnx.Rngs(0)
    import jax.numpy as jnp

    from gemma_4_sql.backends.jax.gemma4.audio import Gemma4AudioModel

    model = Gemma4AudioModel(config, rngs=rngs)
    input_features = jnp.ones((2, 8, 4))
    mask = jnp.ones((2, 8))
    out = model(input_features, mask)
    assert len(out.shape) == 3 and out.shape[:2] == (2, 2)
