"""Decoder layer for Gemma 4."""

from __future__ import annotations

import torch
from torch import nn

from .attention import Gemma4Attention
from .cache import Cache
from .config import Gemma4Config
from .layers import Gemma4MLP, Gemma4RMSNorm
from .moe import Gemma4MoE


class Gemma4DecoderLayer(nn.Module):
    """Decoder layer for Gemma 4."""

    mlp: Gemma4MLP | Gemma4MoE
    pre_feedforward_layernorm: Gemma4RMSNorm | None
    post_feedforward_layernorm: Gemma4RMSNorm | None
    per_layer_input_gate: nn.Linear | None
    per_layer_projection: nn.Linear | None
    post_per_layer_input_norm: Gemma4RMSNorm | None

    def __init__(self, config: Gemma4Config, layer_idx: int) -> None:
        """Initialize Gemma4DecoderLayer.

        Args:
            config: Model configuration parameters.
            layer_idx: Zero-based decoder layer index.

        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.use_post_norms = bool(getattr(config, "use_post_norms", False))
        self.self_attn = Gemma4Attention(config=config, layer_idx=layer_idx)

        if config.num_experts > 1:
            self.mlp = Gemma4MoE(config)
        else:
            self.mlp = Gemma4MLP(config)

        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_post_norms:
            self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.pre_feedforward_layernorm = None
            self.post_feedforward_layernorm = None

        ple_dim = getattr(config, "hidden_size_per_layer_input", None)
        if ple_dim:
            self.per_layer_input_gate = nn.Linear(config.hidden_size, int(ple_dim), bias=False)
            self.per_layer_projection = nn.Linear(int(ple_dim), config.hidden_size, bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

        self.layer_scalar = nn.Parameter(torch.ones((1,)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | Cache | None = None,
        per_layer_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | Cache | None, torch.Tensor | None]:
        """Forward pass for decoder layer.

        Args:
            hidden_states: Input hidden state tensor of shape `(batch, seq_len, hidden_size)`.
            attention_mask: Optional attention mask tensor.
            position_ids: Optional token position index tensor.
            past_key_value: Optional KV cache tuple or `Cache` instance.
            per_layer_input: Optional per-layer embedding input tensor of shape `(batch, seq_len, ple_dim)`.

        Returns:
            Tuple of `(hidden_states, present_key_value, router_logits)`.

        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        if self.use_post_norms:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        if self.use_post_norms and self.pre_feedforward_layernorm is not None:
            normed_ffn = self.pre_feedforward_layernorm(hidden_states)
        else:
            normed_ffn = self.post_attention_layernorm(hidden_states)

        if isinstance(self.mlp, Gemma4MoE):
            mlp_out, router_logits = self.mlp(normed_ffn, original_hidden_states=residual)
        else:
            mlp_out = self.mlp(normed_ffn)
            router_logits = None

        if self.use_post_norms and self.post_feedforward_layernorm is not None:
            mlp_out = self.post_feedforward_layernorm(mlp_out)

        hidden_states = residual + mlp_out

        if per_layer_input is not None and self.per_layer_input_gate is not None and self.per_layer_projection is not None and self.post_per_layer_input_norm is not None:
            ple_residual = hidden_states
            gated = torch.sigmoid(self.per_layer_input_gate(hidden_states)) * per_layer_input
            proj = self.per_layer_projection(gated)
            hidden_states = ple_residual + self.post_per_layer_input_norm(proj)

        hidden_states = hidden_states * self.layer_scalar.to(hidden_states.dtype)

        return hidden_states, present_key_value, router_logits
