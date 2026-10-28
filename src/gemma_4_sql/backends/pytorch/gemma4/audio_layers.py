"""Audio Conformer encoder layers and feature extraction for Gemma 4 in PyTorch."""

from __future__ import annotations

import torch
from torch import nn

from .config import Gemma4AudioConfig
from .layers import Gemma4RMSNorm


class Gemma4AudioFeatureExtractor(nn.Module):
    """Audio feature extractor supporting both 2D raw waveforms and 3D mel spectrograms."""

    def __init__(self, config: Gemma4AudioConfig) -> None:
        """Initialize Gemma4AudioFeatureExtractor.

        Args:
            config: Audio encoder configuration.

        """
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.conv1 = nn.Conv1d(1, self.hidden_size, kernel_size=10, stride=5, bias=False)
        self.conv2 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=2, bias=False)
        self.activation = nn.GELU(approximate="tanh")
        self.mel_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.mel_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.mel_norm = Gemma4RMSNorm(self.hidden_size, eps=float(config.rms_norm_eps))

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Forward pass for audio feature extractor.

        Args:
            input_values: Input waveform tensor of shape `(batch, samples)`, `(batch, 1, samples)`,
                or 3D log-mel spectrogram tensor of shape `(batch, frames, mel_bins)`.

        Returns:
            Extracted audio feature embeddings of shape `(batch, out_frames, hidden_size)`.

        """
        if input_values.dim() == 2:
            waveform = input_values.unsqueeze(1)
            hidden_states = self.activation(self.conv1(waveform))
            hidden_states = self.activation(self.conv2(hidden_states))
            res: torch.Tensor = hidden_states.transpose(1, 2)
            return res

        if input_values.dim() == 3 and input_values.shape[1] == 1:
            hidden_states = self.activation(self.conv1(input_values))
            hidden_states = self.activation(self.conv2(hidden_states))
            res_ch1: torch.Tensor = hidden_states.transpose(1, 2)
            return res_ch1

        # 3D log-mel spectrogram: (batch, frames, mel_bins) -> 2D subsampling convs
        spec = input_values.unsqueeze(1)
        spec_out = self.activation(self.mel_conv1(spec))
        spec_out = self.activation(self.mel_conv2(spec_out))
        bsz, channels, out_frames, out_bins = spec_out.shape
        flattened = spec_out.permute(0, 2, 3, 1).reshape(bsz, out_frames, channels * out_bins)
        # Deterministic adaptive linear projection to self.hidden_size
        if flattened.shape[-1] != self.hidden_size:
            projected = nn.functional.adaptive_avg_pool1d(flattened, self.hidden_size)
        else:
            projected = flattened
        normed_out: torch.Tensor = self.mel_norm(projected)
        return normed_out


class Gemma4ConformerLightConv1d(nn.Module):
    """Conformer LightConv1d block with GLU gating and causal depthwise temporal convolution."""

    def __init__(self, hidden_size: int, kernel_size: int = 5, eps: float = 1e-6) -> None:
        """Initialize Gemma4ConformerLightConv1d.

        Args:
            hidden_size: Audio encoder hidden dimension.
            kernel_size: Depthwise 1D convolution kernel size.
            eps: RMSNorm epsilon.

        """
        super().__init__()
        self.kernel_size = kernel_size
        self.left_pad = max(0, kernel_size - 1)
        self.pre_layer_norm = Gemma4RMSNorm(hidden_size, eps=eps)
        self.linear_start = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.depthwise_conv1d = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=0,
            groups=hidden_size,
            bias=False,
        )
        self.conv_norm = Gemma4RMSNorm(hidden_size, eps=eps)
        self.linear_end = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply GLU-gated causal depthwise 1D convolution with residual connection.

        Args:
            hidden_states: Input tensor of shape `(batch, seq_len, hidden_size)`.

        Returns:
            Convolved output tensor of shape `(batch, seq_len, hidden_size)`.

        """
        residual = hidden_states
        x = self.pre_layer_norm(hidden_states)
        x = self.linear_start(x)
        val, gate = torch.chunk(x, 2, dim=-1)
        x = val * torch.sigmoid(gate)
        x_t = nn.functional.pad(x.transpose(1, 2), (self.left_pad, 0))
        x = self.depthwise_conv1d(x_t).transpose(1, 2)
        x = self.conv_norm(x)
        x = nn.functional.silu(x)
        x = self.linear_end(x)
        res: torch.Tensor = residual + x
        return res


class Gemma4AudioEncoderBlock(nn.Module):
    """Macaron-style Conformer audio encoder block for Gemma 4."""

    def __init__(self, config: Gemma4AudioConfig) -> None:
        """Initialize Gemma4AudioEncoderBlock.

        Args:
            config: Audio encoder configuration.

        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.residual_weight = float(config.residual_weight)
        kernel_size = int(config.conv_kernel_size)

        self.ffw_layer_start = nn.Sequential(
            Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps),
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.lconv1d = Gemma4ConformerLightConv1d(self.hidden_size, kernel_size=kernel_size, eps=config.rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        self.layer_norm2 = nn.LayerNorm(self.hidden_size)
        self.final_norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    @property
    def conv1d(self) -> nn.Conv1d:
        """Provide the underlying causal depthwise 1D convolution layer."""
        return self.lconv1d.depthwise_conv1d

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for Macaron-style Conformer audio encoder block.

        Args:
            hidden_states: Input audio frame embeddings of shape `(batch, seq_len, hidden_size)`.

        Returns:
            Encoded audio representations of shape `(batch, seq_len, hidden_size)`.

        """
        hidden_states = hidden_states + self.residual_weight * self.ffw_layer_start(hidden_states)

        residual = hidden_states
        normed = self.layer_norm1(hidden_states)
        attn_output, _ = self.self_attn(
            query=normed,
            key=normed,
            value=normed,
        )
        hidden_states = residual + attn_output

        hidden_states = self.lconv1d(hidden_states)

        residual = hidden_states
        normed2 = self.layer_norm2(hidden_states)
        mlp_output = self.mlp(normed2)
        out: torch.Tensor = self.final_norm(residual + self.residual_weight * mlp_output)

        return out
