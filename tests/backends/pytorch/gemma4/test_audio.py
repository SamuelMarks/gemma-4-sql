"""Tests for Audio modules."""

import torch

from gemma_4_sql.backends.pytorch.gemma4.audio import Gemma4AudioModel
from gemma_4_sql.backends.pytorch.gemma4.audio_attention import Gemma4AudioCrossAttention
from gemma_4_sql.backends.pytorch.gemma4.audio_layers import Gemma4AudioEncoderBlock, Gemma4AudioFeatureExtractor
from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4AudioConfig, Gemma4Config


def test_gemma4_audio_feature_extractor() -> None:
    """Test Gemma4AudioFeatureExtractor on 2D waveforms, 3D channel-1 waveforms, and 3D mel spectrograms."""
    config = Gemma4AudioConfig(hidden_size=256)
    extractor = Gemma4AudioFeatureExtractor(config)
    x = torch.randn(2, 1000)
    out = extractor(x)
    assert out.shape[0] == 2
    assert out.shape[2] == 256

    x_3d_ch1 = torch.randn(2, 1, 1000)
    out_ch1 = extractor(x_3d_ch1)
    assert out_ch1.shape == out.shape

    # 3D log-mel spectrogram (batch, frames, mel_bins) where channels * out_bins != hidden_size
    x_mel = torch.randn(2, 40, 80)
    out_mel = extractor(x_mel)
    assert out_mel.shape == (2, 10, 256)

    # 3D log-mel spectrogram where channels * out_bins == hidden_size (32 * 20 = 640)
    config_exact = Gemma4AudioConfig(hidden_size=640)
    extractor_exact = Gemma4AudioFeatureExtractor(config_exact)
    out_exact = extractor_exact(torch.randn(2, 40, 80))
    assert out_exact.shape == (2, 10, 640)


def test_gemma4_audio_encoder_block() -> None:
    """Test Gemma4AudioEncoderBlock including Conformer depthwise conv1d gradient propagation."""
    config = Gemma4AudioConfig(hidden_size=256, num_attention_heads=4)
    block = Gemma4AudioEncoderBlock(config)
    x = torch.randn(2, 50, 256, requires_grad=True)
    out = block(x)
    assert out.shape == (2, 50, 256)
    out.sum().backward()
    assert block.conv1d.weight.grad is not None
    assert float(block.conv1d.weight.grad.abs().sum().item()) > 0.0


def test_gemma4_audio_model() -> None:
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


def test_gemma4_audio_cross_attention() -> None:
    """Test Gemma4AudioCrossAttention including non-divisible hidden_size and audio_hidden_size."""
    config = Gemma4Config(hidden_size=512, num_attention_heads=6, audio_config={"hidden_size": 250})
    attn = Gemma4AudioCrossAttention(config)
    hidden = torch.randn(2, 10, 512)
    audio = torch.randn(2, 20, 250)
    out = attn(hidden, audio)
    assert out.shape == (2, 10, 512)
