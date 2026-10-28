"""Audio attention for Gemma 4."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import Gemma4Config


class Gemma4AudioCrossAttention(nn.Module):
    """Audio cross-attention mechanism for Gemma 4."""

    def __init__(self, config: Gemma4Config) -> None:
        """Initialize Gemma4AudioCrossAttention.

        Args:
            config: Gemma 4 model configuration containing `hidden_size`, `num_attention_heads`, and `audio_config`.

        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.audio_hidden_size = config.audio_config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = max(1, self.hidden_size // self.num_heads)
        self.attn_dim = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(self.audio_hidden_size, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(self.audio_hidden_size, self.attn_dim, bias=False)
        self.o_proj = nn.Linear(self.attn_dim, self.hidden_size, bias=False)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=self.num_heads,
            kdim=self.attn_dim,
            vdim=self.attn_dim,
            batch_first=True,
        )

    def forward(self, hidden_states: torch.Tensor, audio_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for audio cross-attention.

        Args:
            hidden_states: Query hidden states of shape `(batch, q_len, hidden_size)`.
            audio_states: Key/value encoded audio states of shape `(batch, kv_len, audio_hidden_size)`.

        Returns:
            Cross-attended output tensor of shape `(batch, q_len, hidden_size)`.

        """
        q = self.q_proj(hidden_states)
        k = self.k_proj(audio_states)
        v = self.v_proj(audio_states)

        attn_output, _ = self.self_attn(
            query=q,
            key=k,
            value=v,
        )
        res: torch.Tensor = self.o_proj(attn_output / math.sqrt(1.0))
        return res
