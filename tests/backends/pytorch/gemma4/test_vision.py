"""Tests for Vision modules."""

import torch

from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4VisionConfig
from gemma_4_sql.backends.pytorch.gemma4.vision import Gemma4VisionEmbeddings, Gemma4VisionModel


def test_gemma4_vision_embeddings():
    """Test Gemma4VisionEmbeddings."""
    config = Gemma4VisionConfig(hidden_size=256, patch_size=14, image_size=224)
    embeddings = Gemma4VisionEmbeddings(config)
    x = torch.randn(2, 3, 224, 224)
    out = embeddings(x)

    num_patches = (224 // 14) ** 2
    assert out.shape == (2, num_patches, 256)


def test_gemma4_vision_model():
    """Test Gemma4VisionModel."""
    config = Gemma4VisionConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4, patch_size=14, image_size=224)
    model = Gemma4VisionModel(config)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    num_patches = (224 // 14) ** 2
    assert out.shape == (2, num_patches, 256)
