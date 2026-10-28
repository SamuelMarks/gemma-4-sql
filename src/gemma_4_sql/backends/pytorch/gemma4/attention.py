"""Attention for Gemma 4."""

from __future__ import annotations

import math

import torch
from torch import nn

from .cache import Cache
from .config import Gemma4Config
from .layers import Gemma4RMSNorm
from .rope import Gemma4RotaryEmbedding, apply_rotary_pos_emb


class Gemma4Attention(nn.Module):
    """Multi-Head / Grouped-Query Attention mechanism for Gemma 4."""

    v_proj: nn.Linear | None
    sliding_window: int | None
    attn_logits_soft_cap: float | None

    def __init__(self, config: Gemma4Config, layer_idx: int) -> None:
        """Initialize Gemma4Attention.

        Args:
            config: Model configuration parameters.
            layer_idx: Zero-based decoder layer index.

        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.sliding_window = config.sliding_window
        global_layers = config.global_attn_layers
        self.is_global = bool(layer_idx in global_layers or config.is_global)

        if self.is_global:
            global_kv = config.num_global_key_value_heads
            self.num_key_value_heads = int(global_kv) if global_kv is not None else int(config.num_key_value_heads)
            global_hd = config.global_head_dim
            self.head_dim = int(global_hd) if global_hd is not None else int(config.head_dim)
            self.share_kv = bool(config.share_kv_projections)
            rope_factor = float(config.global_rope_proportion)
            global_theta = config.global_rope_max_timescale
            rope_base = float(global_theta) if global_theta is not None else float(config.rope_theta)
        else:
            self.num_key_value_heads = int(config.num_key_value_heads)
            self.head_dim = int(config.head_dim)
            self.share_kv = False
            rope_factor = float(config.local_rope_proportion)
            local_theta = config.local_rope_max_timescale
            rope_base = float(local_theta) if local_theta is not None else float(config.rope_theta)

        self.attn_logits_soft_cap = config.attn_logits_soft_cap
        rms_eps = float(config.rms_norm_eps)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        if not self.share_kv:
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        else:
            self.v_proj = None
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=rms_eps, with_scale=True)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=rms_eps, with_scale=True)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=rms_eps, with_scale=False)

        self.rotary_emb = Gemma4RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.sliding_window * 2 if self.sliding_window else 8192,
            base=rope_base,
            factor=rope_factor,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | Cache | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for attention.

        Args:
            hidden_states: Input hidden state tensor of shape `(batch, seq_len, hidden_size)`.
            attention_mask: Optional 2D or 4D attention mask tensor.
            position_ids: Optional token position index tensor of shape `(batch, seq_len)`.
            past_key_value: Optional KV cache tuple or `Cache` instance.

        Returns:
            Tuple containing attention output states and updated key-value cache.

        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_raw = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_raw = key_raw if (self.share_kv or self.v_proj is None) else self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_raw).transpose(1, 2)
        value_states = self.v_norm(value_raw).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if isinstance(past_key_value, tuple):
                kv_seq_len += past_key_value[0].shape[-2]
            else:
                kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        if position_ids is None:
            position_ids = torch.arange(kv_seq_len - q_len, kv_seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            if isinstance(past_key_value, tuple):
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
                past_key_value = (key_states, value_states)
            else:
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
                past_key_value = (key_states, value_states)
        else:
            past_key_value = (key_states, value_states)

        num_key_value_groups = self.num_heads // self.num_key_value_heads
        key_states = key_states[:, :, None, :, :].expand(bsz, self.num_key_value_heads, num_key_value_groups, key_states.shape[2], self.head_dim)
        key_states = key_states.reshape(bsz, self.num_heads, key_states.shape[3], self.head_dim)

        value_states = value_states[:, :, None, :, :].expand(bsz, self.num_key_value_heads, num_key_value_groups, value_states.shape[2], self.head_dim)
        value_states = value_states.reshape(bsz, self.num_heads, value_states.shape[3], self.head_dim)

        is_causal = attention_mask is None and q_len > 1

        if attention_mask is None and self.sliding_window is None and self.attn_logits_soft_cap is None:
            attn_output = nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                is_causal=is_causal,
            )
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if self.attn_logits_soft_cap is not None:
                attn_weights = torch.tanh(attn_weights / self.attn_logits_soft_cap) * self.attn_logits_soft_cap

            if not self.is_global and self.sliding_window is not None:
                min_val = torch.finfo(attn_weights.dtype).min
                sw = self.sliding_window
                window_mask = torch.ones_like(attn_weights, dtype=torch.bool).tril(diagonal=0)
                window_mask = torch.logical_and(window_mask, torch.ones_like(attn_weights, dtype=torch.bool).triu(diagonal=-sw + 1))
                attn_weights = torch.where(window_mask, attn_weights, torch.tensor(min_val, dtype=attn_weights.dtype, device=attn_weights.device))

            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    min_val = torch.finfo(attn_weights.dtype).min
                    key_mask = (attention_mask == 0) if attention_mask.dtype != torch.bool else (~attention_mask)
                    additive_mask = torch.zeros((bsz, 1, 1, kv_seq_len), dtype=attn_weights.dtype, device=attn_weights.device)
                    additive_mask = additive_mask.masked_fill(key_mask[:, None, None, :], min_val)
                    if q_len > 1:
                        causal_mask = torch.ones((q_len, kv_seq_len), dtype=torch.bool, device=attn_weights.device).tril(diagonal=kv_seq_len - q_len)
                        additive_mask = additive_mask.masked_fill(~causal_mask[None, None, :, :], min_val)
                    attn_weights = attn_weights + additive_mask
                else:
                    attn_weights = attn_weights + attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value
