"""Tests for Vision modules."""

import torch

from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4VisionConfig
from gemma_4_sql.backends.pytorch.gemma4.vision import Gemma4VisionEmbeddings, Gemma4VisionModel


def test_gemma4_vision_embeddings() -> None:
    """Test Gemma4VisionEmbeddings."""
    config = Gemma4VisionConfig(hidden_size=256, patch_size=14, image_size=224)
    embeddings = Gemma4VisionEmbeddings(config)
    x = torch.randn(2, 3, 224, 224)
    out = embeddings(x)

    num_patches = (224 // 14) ** 2
    assert out.shape == (2, num_patches, 256)


def test_gemma4_vision_model() -> None:
    """Test Gemma4VisionModel including 2D RoPE and spatial pooling."""
    config = Gemma4VisionConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4, patch_size=14, image_size=224)
    model = Gemma4VisionModel(config)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    num_patches = (224 // 14) ** 2
    assert out.shape == (2, num_patches, 256)

    config_pooled = Gemma4VisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        patch_size=14,
        image_size=56,
        pooling_kernel_size=2,
    )
    model_pooled = Gemma4VisionModel(config_pooled)
    out_pooled = model_pooled(torch.randn(1, 3, 56, 56))
    # 56 // 14 = 4x4 = 16 patches -> pooled 2x2 -> 4 tokens
    assert out_pooled.shape == (1, 4, 64)
    # Non-divisible token count fallback in _pool_image_tokens
    non_square = torch.randn(1, 5, 64)
    assert model_pooled._pool_image_tokens(non_square).shape == (1, 5, 64)
    from gemma_4_sql.backends.pytorch.gemma4.vision import Gemma4VisionAttention

    assert Gemma4VisionAttention(config_pooled)(non_square).shape == (1, 5, 64)
