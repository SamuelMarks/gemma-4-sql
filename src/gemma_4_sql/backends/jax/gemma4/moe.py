"""Provide module docstring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .layers import Gemma4MLP, Gemma4RMSNorm, _make_linear

if TYPE_CHECKING:
    from .config import ModelConfig


class Gemma4RoutedExperts(nnx.Module):
    """Monolithic MoE expert module vectorizing all routed experts."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.config = config
        e = config.num_experts
        h = config.hidden_size
        i_dim = config.moe_intermediate_size if config.moe_intermediate_size is not None else config.intermediate_size
        self.dtype = config.dtype
        functools = __import__("functools")
        ki1 = functools.partial(jax.nn.initializers.normal(stddev=config.hidden_size ** (-0.5)))
        ki2 = functools.partial(jax.nn.initializers.normal(stddev=config.hidden_size ** (-0.5)))
        self.gate_proj_kernel = nnx.Param(ki1(rngs.params(), (e, h, i_dim)))
        self.up_proj_kernel = nnx.Param(ki1(rngs.params(), (e, h, i_dim)))
        self.down_proj_kernel = nnx.Param(ki2(rngs.params(), (e, i_dim, h)))

    def __call__(self, x: Array, topk_indices: Array, topk_weights: Array) -> Array:
        """Apply the selected experts efficiently.

        Args:
        ----
            x: Input sequence (B, T, H)
            topk_indices: Indices of selected experts (B, T, K)
            topk_weights: Weights for selected experts (B, T, K)

        Returns:
        -------
            Output from the routed experts (B, T, H)

        """
        (b, t, h) = x.shape
        k = topk_indices.shape[-1]
        x_flat = x.reshape(b * t, h)
        idx_flat = topk_indices.reshape(b * t, k)
        w_flat = topk_weights.reshape(b * t, k)
        x_expanded = jnp.expand_dims(jnp.expand_dims(x_flat, 1), 1)
        gate_w = jnp.take(self.gate_proj_kernel[...], idx_flat, axis=0)
        up_w = jnp.take(self.up_proj_kernel[...], idx_flat, axis=0)
        down_w = jnp.take(self.down_proj_kernel[...], idx_flat, axis=0)
        gate_out = jnp.matmul(x_expanded, gate_w)
        up_out = jnp.matmul(x_expanded, up_w)
        act = jax.nn.silu(gate_out) * up_out
        out = jnp.matmul(act, down_w).squeeze(2)
        out = out * jnp.expand_dims(w_flat, 2)
        out = jnp.sum(out, axis=1)
        return out.reshape((b, t, h)).astype(self.dtype)


class Gemma4MoE(nnx.Module):
    """Gemma 4 Mixture of Experts combining routed and shared experts.

    Implements a Top-K routing mechanism for multiple parallel MLPs alongside
    a shared MLP that is always executed.
    """

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.config = config
        self.dtype = config.dtype
        shd = config.shd_cfg
        shared_dim = config.intermediate_size * config.num_shared_experts
        self.shared_experts = Gemma4MLP(config.hidden_size, shared_dim, rngs=rngs, dtype=config.dtype, shd=shd)
        self.pre_forward_scale_2 = nnx.Param(jnp.ones(config.hidden_size, dtype=config.weight_dtype))
        self.gate_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=False, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        gate_dtype = jnp.float32 if config.float32_gate_logits else config.dtype
        self.gate = _make_linear(config.hidden_size, config.num_experts, use_bias=False, dtype=gate_dtype, kernel_metadata={}, bias_metadata={}, rngs=rngs)
        self.per_expert_scale = nnx.Param(jnp.ones(config.num_experts, dtype=config.weight_dtype))
        self.routed_experts = Gemma4RoutedExperts(config, rngs=rngs)
        self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        self.post_feedforward_layernorm_1 = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        self.post_feedforward_layernorm_2 = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)

    @jax.named_scope("gemma4_moe")
    def __call__(self, x: Array, original_x: Array) -> Array:
        """Apply Mixture of Experts with shared and routed execution paths."""
        shared_out = self.shared_experts(x)
        shared_out = self.post_feedforward_layernorm_1(shared_out)
        routed_inputs = self.pre_feedforward_layernorm_2(original_x)
        unscaled_norm = self.gate_norm(original_x)
        root_size = self.config.hidden_size ** (-0.5)
        router_scale = jnp.asarray(self.pre_forward_scale_2[...], dtype=unscaled_norm.dtype)
        gate_inputs = unscaled_norm * root_size * router_scale
        router_logits = self.gate(gate_inputs)
        routing_weights = jax.nn.softmax(router_logits, axis=-1)
        (topk_weights, topk_indices) = jax.lax.top_k(routing_weights, k=self.config.num_experts_per_tok)
        topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
        per_expert = jnp.asarray(self.per_expert_scale[...], dtype=topk_weights.dtype)
        topk_weights = topk_weights * per_expert[topk_indices]
        topk_weights = topk_weights.astype(self.dtype)
        routed_out = self.routed_experts(routed_inputs, topk_indices, topk_weights)
        routed_out = self.post_feedforward_layernorm_2(routed_out)
        return shared_out + routed_out
