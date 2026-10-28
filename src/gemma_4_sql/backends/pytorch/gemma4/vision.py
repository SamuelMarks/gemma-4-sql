"""Vision modules for Gemma 4 with 2D Rotary Positional Embeddings and spatial pooling."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import Gemma4VisionConfig
from .layers import Gemma4RMSNorm
from .rope import Gemma4RotaryEmbedding2D, rotate_half


class Gemma4VisionEmbeddings(nn.Module):
    """Vision embeddings for Gemma 4."""

    def __init__(self, config: Gemma4VisionConfig) -> None:
        """Initialize Gemma4VisionEmbeddings.

        Args:
            config: Vision transformer configuration.

        """
        super().__init__()
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.hidden_size = config.hidden_size

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass for vision embeddings.

        Args:
            pixel_values: Input pixel tensor of shape `(batch, 3, height, width)`.

        Returns:
            Patch embeddings tensor of shape `(batch, num_patches, hidden_size)`.

        """
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        positions = torch.arange(self.num_patches, device=pixel_values.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(positions)

        res: torch.Tensor = patch_embeds + position_embeds
        return res


class Gemma4VisionAttention(nn.Module):
    """Vision self-attention for Gemma 4 with 2D Rotary Positional Embeddings."""

    def __init__(self, config: Gemma4VisionConfig) -> None:
        """Initialize Gemma4VisionAttention.

        Args:
            config: Vision transformer configuration.

        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.rotary_emb_2d = Gemma4RotaryEmbedding2D(dim=self.head_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for vision attention with 2D RoPE when head_dim >= 4.

        Args:
            hidden_states: Input patch embeddings of shape `(batch, seq_len, hidden_size)`.

        Returns:
            Output projected attention tensor of shape `(batch, seq_len, hidden_size)`.

        """
        bsz, seq_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        grid_side = math.isqrt(seq_len)
        if grid_side * grid_side == seq_len and self.head_dim >= 4 and self.head_dim % 4 == 0:
            cos, sin = self.rotary_emb_2d(hidden_states, grid_side, grid_side)
            cos_b = cos.unsqueeze(0).unsqueeze(0)
            sin_b = sin.unsqueeze(0).unsqueeze(0)
            q = (q * cos_b) + (rotate_half(q) * sin_b)
            k = (k * cos_b) + (rotate_half(k) * sin_b)

        attn_output = nn.functional.scaled_dot_product_attention(q, k, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        res: torch.Tensor = self.o_proj(attn_output)
        return res


class Gemma4VisionEncoderLayer(nn.Module):
    """Vision encoder layer for Gemma 4."""

    def __init__(self, config: Gemma4VisionConfig) -> None:
        """Initialize Gemma4VisionEncoderLayer.

        Args:
            config: Vision transformer configuration.

        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma4VisionAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, config.intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.intermediate_size, self.hidden_size),
        )
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for vision encoder layer.

        Args:
            hidden_states: Input hidden states tensor.

        Returns:
            Encoded layer hidden states tensor.

        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        out: torch.Tensor = residual + hidden_states

        return out


class Gemma4VisionModel(nn.Module):
    """Vision model for Gemma 4 with optional 2D spatial token pooling."""

    def __init__(self, config: Gemma4VisionConfig) -> None:
        """Initialize Gemma4VisionModel.

        Args:
            config: Vision transformer configuration.

        """
        super().__init__()
        self.config = config
        self.embeddings = Gemma4VisionEmbeddings(config)
        self.layers = nn.ModuleList([Gemma4VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.post_layernorm = Gemma4RMSNorm(config.hidden_size)
        self.pooling_kernel_size = int(config.pooling_kernel_size or 1)

    def _pool_image_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply 2D spatial average pooling over vision patch tokens when `pooling_kernel_size > 1`.

        Args:
            tokens: Patch embeddings of shape `(batch, seq_len, hidden_size)`.

        Returns:
            Spatially pooled patch embeddings.

        """
        k = self.pooling_kernel_size
        if k <= 1:
            return tokens
        bsz, seq_len, dim = tokens.shape
        side = math.isqrt(seq_len)
        if side * side != seq_len or side % k != 0:
            return tokens
        grid = tokens.view(bsz, side // k, k, side // k, k, dim)
        pooled: torch.Tensor = grid.mean(dim=(2, 4)).reshape(bsz, (side // k) ** 2, dim)
        return pooled

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass for vision model.

        Args:
            pixel_values: Input pixel tensor of shape `(batch, 3, height, width)`.

        Returns:
            Final vision representations tensor.

        """
        hidden_states = self.embeddings(pixel_values)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.post_layernorm(hidden_states)
        res: torch.Tensor = self._pool_image_tokens(hidden_states)
        return res
