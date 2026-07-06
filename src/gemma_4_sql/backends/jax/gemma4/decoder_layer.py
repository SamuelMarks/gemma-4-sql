"""Gemma 4 Decoder Layer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .attention import Gemma4Attention
from .layers import Gemma4MLP, Gemma4RMSNorm, _make_linear
from .moe import Gemma4MoE

if TYPE_CHECKING:
    from .config import AttentionType, ModelConfig
    from .modeling import LayerCache


class Gemma4DecoderLayer(nnx.Module):
    """A single decoder layer combining Attention, MoE, and Normalization."""

    def __init__(self, config: ModelConfig, attention_type: AttentionType, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__.

        Args:
            config: The configuration parameters.
            attention_type: The attention type.
            rngs: The rngs.
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
            self.per_layer_input_gate = _make_linear(config.hidden_size, config.hidden_size_per_layer_input, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
            self.per_layer_projection = _make_linear(config.hidden_size_per_layer_input, config.hidden_size, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
            self.post_per_layer_input_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        self.layer_scalar = nnx.Param(jnp.ones(1, dtype=config.weight_dtype))

    @jax.named_scope("gemma4_layer")
    def __call__(self, x: Array, positions: Array, layer_cache: LayerCache | None = None, attention_mask: Array | None = None, per_layer_input: Array | None = None) -> Array:
        """Apply the decoder layer.

        Returns:
            object: The resulting output from the operation.

        """
        lnx = self.pre_self_attention_norm(x)
        attn_out = self.self_attention(lnx, positions, layer_cache, attention_mask)
        attn_out = self.post_self_attention_norm(attn_out)
        x += attn_out
        lnx2 = self.pre_ffw_norm(x)
        mlp_out = self.mlp(lnx2, original_x=x)
        mlp_out = self.post_ffw_norm(mlp_out)
        x += mlp_out
        if self.config.hidden_size_per_layer_input and per_layer_input is not None:
            residual = x
            x_ple = self.per_layer_input_gate(x)
            ple_out = jax.nn.sigmoid(x_ple) * per_layer_input
            ple_proj = self.per_layer_projection(ple_out)
            x = residual + self.post_per_layer_input_norm(ple_proj)
        return x
