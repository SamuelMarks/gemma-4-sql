"""Tests for Audio modules."""

import torch

from gemma_4_sql.backends.pytorch.gemma4.audio import Gemma4AudioModel
from gemma_4_sql.backends.pytorch.gemma4.audio_attention import Gemma4AudioCrossAttention
from gemma_4_sql.backends.pytorch.gemma4.audio_layers import Gemma4AudioEncoderBlock, Gemma4AudioFeatureExtractor
from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4AudioConfig, Gemma4Config


def test_gemma4_audio_feature_extractor():
    """Test Gemma4AudioFeatureExtractor."""
    config = Gemma4AudioConfig(hidden_size=256)
    extractor = Gemma4AudioFeatureExtractor(config)
    x = torch.randn(2, 1000)
    out = extractor(x)
    assert out.shape[0] == 2
    assert out.shape[2] == 256


def test_gemma4_audio_encoder_block():
    """Test Gemma4AudioEncoderBlock."""
    config = Gemma4AudioConfig(hidden_size=256, num_attention_heads=4)
    block = Gemma4AudioEncoderBlock(config)
    x = torch.randn(2, 50, 256)
    out = block(x)
    assert out.shape == (2, 50, 256)


def test_gemma4_audio_model():
    """Test Gemma4AudioModel."""
    config = Gemma4Config(
        hidden_size=512,
        audio_config={"hidden_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4},
    )
    model = Gemma4AudioModel(config)
    x = torch.randn(2, 1000)
    out = model(x)
    assert out.shape[0] == 2
    assert out.shape[2] == 512


def test_gemma4_audio_cross_attention():
    """Test Gemma4AudioCrossAttention."""
    config = Gemma4Config(hidden_size=512, num_attention_heads=4, audio_config={"hidden_size": 256})
    attn = Gemma4AudioCrossAttention(config)
    hidden = torch.randn(2, 10, 512)
    audio = torch.randn(2, 20, 256)
    out = attn(hidden, audio)
    assert out.shape == (2, 10, 512)
