"""Unit and behavioral tests for gemma4 vision."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from flax import nnx

from tests.conftest import requires_jax

pytestmark = requires_jax

from gemma_4_sql.backends.jax.gemma4.config import AudioConfig, ModelConfig, VisionConfig
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


def _make_vision_config() -> VisionConfig:
    """Construct a small VisionConfig for unit tests."""
    return VisionConfig(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=16,
        num_attention_heads=4,
        intermediate_size=32,
        layer_norm_eps=1e-6,
        num_hidden_layers=1,
    )


def _make_audio_config() -> AudioConfig:
    """Construct a small AudioConfig for unit tests."""
    return AudioConfig(
        hidden_size=16,
        output_proj_dims=16,
        rms_norm_eps=1e-6,
    )


def _make_model_config(
    *,
    with_vision: bool = True,
    with_audio: bool = True,
    mm_tokens_per_image: int = 4,
) -> ModelConfig:
    """Construct a small ModelConfig with optional vision and audio configs."""
    cfg = ModelConfig(
        hidden_size=16,
        dtype=jnp.float32,
        vision_config=_make_vision_config() if with_vision else None,
        audio_config=_make_audio_config() if with_audio else None,
        mm_tokens_per_image=mm_tokens_per_image,
    )
    if not with_vision:
        cfg.vision_config = None
    if not with_audio:
        cfg.audio_config = None
    return cfg


def test_siglip_vision_embeddings() -> None:
    """Verify siglip vision embeddings behavior."""
    rngs = nnx.Rngs(0)
    config = _make_vision_config()
    model = SiglipVisionEmbeddings(config, rngs=rngs)

    # 32x32 image with 16x16 patch = 2x2 = 4 patches
    pixel_values = jnp.ones((2, 32, 32, 3))
    out = model(pixel_values)
    assert out.shape == (2, 4, 16)


def test_siglip_attention() -> None:
    """Verify siglip attention behavior."""
    rngs = nnx.Rngs(0)
    config = _make_vision_config()
    model = SiglipAttention(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = model(x)
    assert out.shape == (2, 4, 16)


def test_siglip_mlp() -> None:
    """Verify siglip mlp behavior."""
    rngs = nnx.Rngs(0)
    config = _make_vision_config()
    model = SiglipMLP(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = model(x)
    assert out.shape == (2, 4, 16)


def test_siglip_encoder_layer() -> None:
    """Verify siglip encoder layer behavior."""
    rngs = nnx.Rngs(0)
    config = _make_vision_config()
    model = SiglipEncoderLayer(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = model(x)
    assert out.shape == (2, 4, 16)


def test_gemma4_multimodal_embedder() -> None:
    """Verify gemma4 multimodal embedder behavior."""
    rngs = nnx.Rngs(0)

    config = _make_model_config(with_vision=True, with_audio=True)
    model = Gemma4MultimodalEmbedder(config, rngs=rngs)

    x = jnp.ones((2, 4, 16))
    out = model(x)
    assert out.shape == (2, 4, 16)

    config2 = _make_model_config(with_vision=True, with_audio=False)
    model2 = Gemma4MultimodalEmbedder(config2, rngs=rngs)
    out2 = model2(x)
    assert out2.shape == (2, 4, 16)


def test_siglip_vision_transformer() -> None:
    """Verify siglip vision transformer behavior."""
    rngs = nnx.Rngs(0)
    config = _make_vision_config()
    model = SiglipVisionTransformer(config, rngs=rngs)

    pixel_values = jnp.ones((2, 32, 32, 3))
    out = model(pixel_values)
    assert out.shape == (2, 4, 16)


def test_avg_pool_vision_outputs() -> None:
    """Verify avg pool vision outputs behavior."""
    config = _make_vision_config()
    x = jnp.ones((2, 4, 16))
    out = avg_pool_vision_outputs(x, kernel_size=2, config=config, num_output_tokens=1)

    assert out.shape == (2, 1, 16)
    assert jnp.allclose(out, jnp.ones((2, 1, 16)))


def test_gemma4_multi_modal_projector() -> None:
    """Verify gemma4 multi modal projector behavior."""
    rngs = nnx.Rngs(0)
    config = _make_model_config(with_vision=True, with_audio=True, mm_tokens_per_image=1)

    model = Gemma4MultiModalProjector(config, rngs=rngs)
    x = jnp.ones((2, 4, 16))
    out = model(x)
    assert out.shape == (2, 1, 16)

    config_no_vision = _make_model_config(with_vision=False, with_audio=True)
    with pytest.raises(ValueError, match="Vision config is required"):
        Gemma4MultiModalProjector(config_no_vision, rngs=rngs)
