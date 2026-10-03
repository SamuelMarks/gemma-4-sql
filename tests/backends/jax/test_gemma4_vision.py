import jax.numpy as jnp
import pytest
from flax import nnx

from gemma_4_sql.backends.jax.gemma4.vision import (
    Gemma4MultimodalEmbedder,
    Gemma4MultiModalProjector,
    SiglipAttention,
    SiglipEncoderLayer,
    SiglipMLP,
    SiglipVisionEmbeddings,
    SiglipVisionTransformer,
    avg_pool_vision_outputs,
)


class DummyVisionConfig:
    image_size = 32
    patch_size = 16
    num_channels = 3
    hidden_size = 16
    num_attention_heads = 4
    intermediate_size = 32
    layer_norm_eps = 1e-6
    num_hidden_layers = 1

    class shd_cfg:
        layer_norm = None


class DummyAudioConfig:
    hidden_size = 16
    output_proj_dims = 16
    rms_norm_eps = 1e-6


class DummyConfig:
    hidden_size = 16
    dtype = jnp.float32
    vision_config = DummyVisionConfig()
    audio_config = DummyAudioConfig()
    mm_tokens_per_image = 4


def test_siglip_vision_embeddings():
    rngs = nnx.Rngs(0)
    config = DummyVisionConfig()
    model = SiglipVisionEmbeddings(config, rngs=rngs)

    # 32x32 image with 16x16 patch = 2x2 = 4 patches
    pixel_values = jnp.ones((2, 32, 32, 3))
    out = model(pixel_values)
    assert out.shape == (2, 4, 16)


def test_siglip_attention():
    rngs = nnx.Rngs(0)
    config = DummyVisionConfig()
    model = SiglipAttention(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = model(x)
    assert out.shape == (2, 4, 16)


def test_siglip_mlp():
    rngs = nnx.Rngs(0)
    config = DummyVisionConfig()
    model = SiglipMLP(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = model(x)
    assert out.shape == (2, 4, 16)


def test_siglip_encoder_layer():
    rngs = nnx.Rngs(0)
    config = DummyVisionConfig()
    model = SiglipEncoderLayer(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = model(x)
    assert out.shape == (2, 4, 16)


def test_gemma4_multimodal_embedder():
    rngs = nnx.Rngs(0)

    config = DummyConfig()
    model = Gemma4MultimodalEmbedder(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = model(x)
    assert out.shape == (2, 4, 16)

    config2 = DummyConfig()
    config2.audio_config = None
    model2 = Gemma4MultimodalEmbedder(config2, rngs=rngs)
    out2 = model2(x)
    assert out2.shape == (2, 4, 16)


def test_siglip_vision_transformer():
    rngs = nnx.Rngs(0)
    config = DummyVisionConfig()
    model = SiglipVisionTransformer(config, rngs=rngs)

    pixel_values = jnp.ones((2, 32, 32, 3))
    out = model(pixel_values)
    assert out.shape == (2, 4, 16)


def test_avg_pool_vision_outputs():
    config = DummyVisionConfig()
    # 4 patches. image_size=32, patch_size=16 -> 2 patches per side.
    # tokens_per_side = sqrt(num_output_tokens). Let's use 1 output token.
    x = jnp.ones((2, 4, 16))
    out = avg_pool_vision_outputs(x, kernel_size=2, config=config, num_output_tokens=1)

    assert out.shape == (2, 1, 16)
    assert jnp.allclose(out, jnp.ones((2, 1, 16)))


def test_gemma4_multi_modal_projector():
    rngs = nnx.Rngs(0)
    config = DummyConfig()
    config.mm_tokens_per_image = 1

    model = Gemma4MultiModalProjector(config, rngs=rngs)
    x = jnp.ones((2, 4, 16))
    out = model(x)
    assert out.shape == (2, 1, 16)

    config_no_vision = DummyConfig()
    config_no_vision.vision_config = None
    with pytest.raises(ValueError, match="Vision config is required"):
        Gemma4MultiModalProjector(config_no_vision, rngs=rngs)
