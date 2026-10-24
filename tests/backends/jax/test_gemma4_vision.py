"""Unit and behavioral tests for gemma4 vision."""

from typing import Any

import jax.numpy as jnp
import pytest
from flax import nnx

from tests.conftest import requires_jax

pytestmark = requires_jax

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
    """Test stub for `DummyVisionConfig`."""

    image_size = 32
    patch_size = 16
    num_channels = 3
    hidden_size = 16
    num_attention_heads = 4
    intermediate_size = 32
    layer_norm_eps = 1e-6
    num_hidden_layers = 1

    class shd_cfg:
        """Test stub for `shd_cfg`."""

        layer_norm = None


class DummyAudioConfig:
    """Test stub for `DummyAudioConfig`."""

    hidden_size = 16
    output_proj_dims = 16
    rms_norm_eps = 1e-6


class DummyConfig:
    """Test stub for `DummyConfig`."""

    hidden_size = 16
    dtype = jnp.float32
    vision_config: Any = DummyVisionConfig()
    audio_config: Any = DummyAudioConfig()
    mm_tokens_per_image = 4


def test_siglip_vision_embeddings():
    """Verify siglip vision embeddings behavior."""
    rngs = nnx.Rngs(0)
    config: Any = DummyVisionConfig()
    model = SiglipVisionEmbeddings(config, rngs=rngs)

    # 32x32 image with 16x16 patch = 2x2 = 4 patches
    pixel_values = jnp.ones((2, 32, 32, 3))
    out = model(pixel_values)
    assert out.shape == (2, 4, 16)


def test_siglip_attention():
    """Verify siglip attention behavior."""
    rngs = nnx.Rngs(0)
    config: Any = DummyVisionConfig()
    model = SiglipAttention(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = model(x)
    assert out.shape == (2, 4, 16)


def test_siglip_mlp():
    """Verify siglip mlp behavior."""
    rngs = nnx.Rngs(0)
    config: Any = DummyVisionConfig()
    model = SiglipMLP(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = model(x)
    assert out.shape == (2, 4, 16)


def test_siglip_encoder_layer():
    """Verify siglip encoder layer behavior."""
    rngs = nnx.Rngs(0)
    config: Any = DummyVisionConfig()
    model = SiglipEncoderLayer(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = model(x)
    assert out.shape == (2, 4, 16)


def test_gemma4_multimodal_embedder():
    """Verify gemma4 multimodal embedder behavior."""
    rngs = nnx.Rngs(0)

    config: Any = DummyConfig()
    model = Gemma4MultimodalEmbedder(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = model(x)
    assert out.shape == (2, 4, 16)

    config2: Any = DummyConfig()
    config2.audio_config = None
    model2 = Gemma4MultimodalEmbedder(config2, rngs=rngs)
    out2 = model2(x)
    assert out2.shape == (2, 4, 16)


def test_siglip_vision_transformer():
    """Verify siglip vision transformer behavior."""
    rngs = nnx.Rngs(0)
    config: Any = DummyVisionConfig()
    model = SiglipVisionTransformer(config, rngs=rngs)

    pixel_values = jnp.ones((2, 32, 32, 3))
    out = model(pixel_values)
    assert out.shape == (2, 4, 16)


def test_avg_pool_vision_outputs():
    """Verify avg pool vision outputs behavior."""
    config: Any = DummyVisionConfig()
    # 4 patches. image_size=32, patch_size=16 -> 2 patches per side.
    # tokens_per_side = sqrt(num_output_tokens). Let's use 1 output token.
    x = jnp.ones((2, 4, 16))
    out = avg_pool_vision_outputs(x, kernel_size=2, config=config, num_output_tokens=1)

    assert out.shape == (2, 1, 16)
    assert jnp.allclose(out, jnp.ones((2, 1, 16)))


def test_gemma4_multi_modal_projector():
    """Verify gemma4 multi modal projector behavior."""
    rngs = nnx.Rngs(0)
    config: Any = DummyConfig()
    config.mm_tokens_per_image = 1

    model = Gemma4MultiModalProjector(config, rngs=rngs)
    x = jnp.ones((2, 4, 16))
    out = model(x)
    assert out.shape == (2, 1, 16)

    config_no_vision: Any = DummyConfig()
    config_no_vision.vision_config = None
    with pytest.raises(ValueError, match="Vision config is required"):
        Gemma4MultiModalProjector(config_no_vision, rngs=rngs)
