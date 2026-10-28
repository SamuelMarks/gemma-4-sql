"""Mixture of Experts for Gemma 4."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import Gemma4Config
from .layers import Gemma4MLP, Gemma4RMSNorm


class Gemma4MoERouter(nn.Module):
    """Router for Mixture of Experts with RMS normalization and per-expert scaling."""

    def __init__(self, config: Gemma4Config) -> None:
        """Initialize Gemma4MoERouter.

        Args:
            config: Model configuration parameters.

        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.float32_gate_logits = bool(config.float32_gate_logits)
        self.router_jitter_noise = float(config.router_jitter_noise)
        rms_eps = float(config.rms_norm_eps)

        self.gate_norm = Gemma4RMSNorm(config.hidden_size, eps=rms_eps, with_scale=False)
        self.scale = nn.Parameter(torch.ones(config.hidden_size))
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))
        self.root_size = float(config.hidden_size ** (-0.5))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for the router.

        Args:
            hidden_states: Flattened or 3D token hidden states.

        Returns:
            Tuple of `(routing_weights, selected_experts, router_logits)`.

        """
        if self.training and self.router_jitter_noise > 0:
            jitter = torch.empty_like(hidden_states).uniform_(-self.router_jitter_noise, self.router_jitter_noise)
            hidden_states = hidden_states * (1.0 + jitter)

        normed = self.gate_norm(hidden_states)
        gate_inputs = normed * self.root_size * self.scale.to(normed.dtype)
        router_logits = self.gate(gate_inputs)
        if self.float32_gate_logits:
            router_logits = router_logits.float()
        routing_weights = F.softmax(router_logits, dim=-1)

        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights * self.per_expert_scale.to(routing_weights.dtype)[selected_experts]
        routing_weights = routing_weights.to(hidden_states.dtype)

        return routing_weights, selected_experts, router_logits


def calculate_load_balancing_loss(router_logits: torch.Tensor, num_experts: int, top_k: int) -> torch.Tensor:
    """Calculate the auxiliary load balancing loss.

    Args:
        router_logits: Raw logits from router gate of shape `(num_tokens, num_experts)`.
        num_experts: Total number of experts.
        top_k: Top-k chosen experts count.

    Returns:
        Auxiliary load balancing loss scalar tensor.

    """
    router_probs = F.softmax(router_logits.float(), dim=-1)
    router_probs_mean = router_probs.mean(dim=0)

    _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts)
    expert_mask = expert_mask.sum(dim=1).float()
    expert_mask_mean = expert_mask.mean(dim=0)

    loss = (router_probs_mean * expert_mask_mean).sum() * num_experts
    return loss


class Gemma4MoE(nn.Module):
    """Gemma 4 Mixture of Experts layer combining dense shared experts and sparse routed experts."""

    last_aux_loss: torch.Tensor | None

    def __init__(self, config: Gemma4Config) -> None:
        """Initialize Gemma4MoE.

        Args:
            config: Model configuration parameters.

        """
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        num_shared = int(config.num_shared_experts)
        shared_inter = config.intermediate_size * max(1, num_shared)
        routed_inter = config.moe_intermediate_size if config.moe_intermediate_size is not None else config.intermediate_size
        rms_eps = float(config.rms_norm_eps)

        self.shared_experts = Gemma4MLP(config, intermediate_size=shared_inter)
        self.router = Gemma4MoERouter(config)
        self.experts = nn.ModuleList([Gemma4MLP(config, intermediate_size=routed_inter) for _ in range(self.num_experts)])
        self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(config.hidden_size, eps=rms_eps)
        self.post_feedforward_layernorm_1 = Gemma4RMSNorm(config.hidden_size, eps=rms_eps)
        self.post_feedforward_layernorm_2 = Gemma4RMSNorm(config.hidden_size, eps=rms_eps)
        self.last_aux_loss = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for MoE layer combining shared and top-k routed experts.

        Args:
            hidden_states: Pre-normalized hidden state tensor of shape `(batch, seq_len, hidden_size)`.
            original_hidden_states: Optional unnormalized residual hidden state tensor for routing.

        Returns:
            Tuple of `(combined_hidden_states, router_logits)`.

        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        routing_source = original_hidden_states if original_hidden_states is not None else hidden_states

        shared_out = self.shared_experts(hidden_states)
        shared_out = self.post_feedforward_layernorm_1(shared_out)

        routed_inputs = self.pre_feedforward_layernorm_2(routing_source).view(-1, hidden_dim)
        flat_routing_source = routing_source.view(-1, hidden_dim)

        routing_weights, selected_experts, router_logits = self.router(flat_routing_source)
        self.last_aux_loss = calculate_load_balancing_loss(router_logits, self.num_experts, self.num_experts_per_tok)

        routed_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        active_experts = torch.unique(selected_experts)
        for expert_tensor in active_experts:
            expert_idx = int(expert_tensor.item())
            idx, top_x = torch.where(selected_experts == expert_idx)
            expert_layer = self.experts[expert_idx]
            expert_tokens = routed_inputs[idx]
            expert_weights = routing_weights[idx, top_x].unsqueeze(-1)
            expert_out = expert_layer(expert_tokens) * expert_weights
            routed_hidden_states.index_add_(0, idx, expert_out.to(hidden_states.dtype))

        routed_out = routed_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        routed_out = self.post_feedforward_layernorm_2(routed_out)

        return shared_out + routed_out, router_logits
