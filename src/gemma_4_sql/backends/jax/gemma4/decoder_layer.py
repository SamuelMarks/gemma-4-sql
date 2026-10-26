"""Gemma 4 Decoder Layer implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .attention import Gemma4Attention
from .cache import LayerCache
from .config import AttentionType, ModelConfig
from .layers import Gemma4MLP, Gemma4RMSNorm, make_linear
from .moe import Gemma4MoE


class Gemma4DecoderLayer(nnx.Module):
    """A single decoder layer combining Attention, MoE, and Normalization."""

    mlp: Gemma4MLP | Gemma4MoE

    def __init__(self, config: ModelConfig, attention_type: AttentionType, *, rngs: nnx.Rngs) -> None:
        """Initialize the Gemma 4 decoder layer.

        Args:
            config: Model configuration parameters.
            attention_type: Attention mechanism variant (`GLOBAL` or `LOCAL_SLIDING`).
            rngs: Flax NNX random number generators.

        """
        self.config = config
        shd = config.shd_cfg
        self.pre_self_attention_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        self.self_attention = Gemma4Attention(config, attention_type, rngs=rngs)
        self.post_self_attention_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        self.pre_ffw_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        if config.num_experts > 1:
            self.mlp = Gemma4MoE(config, rngs=rngs)
        else:
            self.mlp = Gemma4MLP(config.hidden_size, config.intermediate_size, rngs=rngs, dtype=config.dtype, shd=shd)
        self.post_ffw_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        if config.hidden_size_per_layer_input:
            self.per_layer_input_gate = make_linear(config.hidden_size, config.hidden_size_per_layer_input, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
            self.per_layer_projection = make_linear(config.hidden_size_per_layer_input, config.hidden_size, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
            self.post_per_layer_input_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        self.layer_scalar = nnx.Param(jnp.ones(1, dtype=config.weight_dtype))

    @jax.named_scope("gemma4_layer")
    def __call__(self, x: Array, positions: Array, layer_cache: LayerCache | None = None, attention_mask: Array | None = None, per_layer_input: Array | None = None) -> Array:
        """Apply the decoder layer.

        Args:
            x: Input hidden state tensor.
            positions: Token position indices tensor.
            layer_cache: Optional KV cache for this decoder layer.
            attention_mask: Optional additive attention mask.
            per_layer_input: Optional per-layer embedding input tensor.

        Returns:
            Output hidden state tensor after self-attention, MLP/MoE, and PLE gating.

        """
        lnx = self.pre_self_attention_norm(x)
        attn_out = self.self_attention(lnx, positions, layer_cache, attention_mask)
        if getattr(self.config, "use_post_norms", True):
            attn_out = self.post_self_attention_norm(attn_out)
        x += attn_out
        lnx2 = self.pre_ffw_norm(x)
        mlp_out = self.mlp(lnx2, original_x=x)
        if getattr(self.config, "use_post_norms", True):
            mlp_out = self.post_ffw_norm(mlp_out)
        x += mlp_out
        if self.config.hidden_size_per_layer_input and per_layer_input is not None:
            residual = x
            x_ple = self.per_layer_input_gate(x)
            ple_out = jax.nn.sigmoid(x_ple) * per_layer_input
            ple_proj = self.per_layer_projection(ple_out)
            x = residual + self.post_per_layer_input_norm(ple_proj)
        x = x * jnp.asarray(self.layer_scalar[...], dtype=x.dtype)
        return x
