"""Gemma 4 Attention implementation."""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .cache import LayerCache
from .config import AttentionType, ModelConfig
from .layers import Gemma4RMSNorm, LinearLayerProtocol, make_linear
from .rope import RoPE, apply_rope

MASK_PENALTY = -10000.0


def _compute_attention_scores_and_output(qkv: tuple[jax.Array, jax.Array, jax.Array], attention_mask: jax.Array | None, config_params: tuple[int, float | None, int, int]) -> jax.Array:
    """Compute scaled dot-product attention scores, optional soft-capping, and context projection.

    Args:
        qkv: A 3-tuple of `(query, key, value)` tensors of shapes `(B, T, H_q, D)` and `(B, S, H_kv, D)`.
        attention_mask: Optional additive attention mask tensor of shape `(B, 1, T, S)`.
        config_params: A 4-tuple of `(head_dim, attn_logits_soft_cap, num_kv_heads, num_heads)`.

    Returns:
        Flattened attention output tensor of shape `(B, T, num_heads * head_dim)`.

    """
    (q, k, v) = qkv
    (head_dim, soft_cap, num_kv_heads, num_heads) = config_params
    scale = 1.0 / math.sqrt(head_dim)
    if num_kv_heads != num_heads:
        num_rep = num_heads // num_kv_heads
        k = jnp.repeat(k, num_rep, axis=2)
        v = jnp.repeat(v, num_rep, axis=2)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if soft_cap is not None:
        scores /= soft_cap
        scores = jnp.tanh(scores) * soft_cap
    if attention_mask is not None:
        scores = scores + attention_mask
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return out.reshape((out.shape[0], out.shape[1], -1))


def _prepare_qkv_for_attention(qkv: tuple[jax.Array, jax.Array, jax.Array], positions: Array, rope: RoPE, cache: LayerCache | None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Apply Rotary Position Embeddings (RoPE), update the KV cache, and construct causal/window masks.

    Args:
        qkv: A 3-tuple of `(query, key, value)` tensors of shape `(B, T, H, D)`.
        positions: Token position index tensor of shape `(B, T)`.
        rope: `RoPE` module callable returning `(sin, cos)` frequency tables.
        cache: Optional mutable `LayerCache` for autoregressive decoding.

    Returns:
        A 5-tuple of `(q_rotated, k_cached, v_cached, causal_mask, relative_window_distance)`.

    """
    (q, k, v) = qkv
    (_batch_size, seq_len, _, _) = q.shape
    (sin, cos) = rope(positions)
    q = apply_rope(q, sin, cos)
    k = apply_rope(k, sin, cos)
    if cache is not None:
        slice_indices = (0, cache.cur_ind.value, 0, 0)
        cache.k_cache.value = jax.lax.dynamic_update_slice(cache.k_cache[...], k, slice_indices)
        cache.v_cache.value = jax.lax.dynamic_update_slice(cache.v_cache[...], v, slice_indices)
        k = cache.k_cache[...]
        v = cache.v_cache[...]
        mask_len = k.shape[1]
    else:
        mask_len = seq_len
    mask = jnp.arange(mask_len)[None, :] <= positions[:, :, None]
    window = positions[:, :, None] - jnp.arange(mask_len)[None, :]
    return (q, k, v, mask, window)


def _is_global_attention(attention_type: AttentionType | str) -> bool:
    """Return True if `attention_type` represents global attention."""
    if isinstance(attention_type, AttentionType):
        return attention_type == AttentionType.GLOBAL
    return str(attention_type).upper() in ("GLOBAL", "ATTENTIONTYPE.GLOBAL")


def _is_local_sliding_attention(attention_type: AttentionType | str) -> bool:
    """Return True if `attention_type` represents local sliding-window attention."""
    if isinstance(attention_type, AttentionType):
        return attention_type == AttentionType.LOCAL_SLIDING
    return str(attention_type).upper() in ("LOCAL_SLIDING", "ATTENTIONTYPE.LOCAL_SLIDING")


class Gemma4Attention(nnx.Module):
    """Multi-Head / Grouped-Query Attention for Gemma 4."""

    q_proj: LinearLayerProtocol
    k_proj: LinearLayerProtocol
    v_proj: LinearLayerProtocol | None
    o_proj: LinearLayerProtocol
    q_norm: Gemma4RMSNorm
    k_norm: Gemma4RMSNorm
    v_norm: Gemma4RMSNorm

    def _setup_heads(self, config: ModelConfig, attention_type: AttentionType | str) -> None:
        """Configure attention head counts, head dimensions, and KV sharing flags.

        Args:
            config: Model configuration parameters.
            attention_type: Whether this layer uses global or local sliding attention.

        """
        self.num_heads = config.num_attention_heads
        if _is_global_attention(attention_type):
            self.num_kv_heads = config.num_global_key_value_heads if config.num_global_key_value_heads is not None else config.num_key_value_heads
            self.head_dim = config.global_head_dim if config.global_head_dim is not None else config.head_dim
            self.share_kv = config.share_kv_projections
        else:
            self.num_kv_heads = config.num_key_value_heads
            self.head_dim = config.head_dim
            self.share_kv = False

    def _setup_rope(self, config: ModelConfig, attention_type: AttentionType | str) -> None:
        """Configure Rotary Position Embedding (RoPE) frequencies for the layer.

        Args:
            config: Model configuration parameters.
            attention_type: Whether this layer uses global or local sliding attention.

        """
        if _is_global_attention(attention_type):
            rope_factor = float(config.global_rope_proportion)
            raw_theta = config.global_rope_max_timescale
        else:
            rope_factor = float(config.local_rope_proportion)
            raw_theta = config.local_rope_max_timescale
        rope_theta = float(raw_theta if raw_theta is not None else config.rope_max_timescale)
        self.rope = RoPE(rope_type="default", head_dim=self.head_dim, rope_theta=int(rope_theta), factor=rope_factor)

    def __init__(self, config: ModelConfig, attention_type: AttentionType | str, *, rngs: nnx.Rngs) -> None:
        """Initialize Multi-Head / Grouped-Query Attention module.

        Args:
            config: Model configuration parameters.
            attention_type: Attention mechanism variant (`GLOBAL` or `LOCAL_SLIDING`).
            rngs: Flax NNX random number generators.

        """
        self.config = config
        self.attention_type = attention_type
        self._setup_heads(config, attention_type)
        self.hidden_size = config.hidden_size
        self.dtype = config.dtype
        shd = config.shd_cfg
        self.q_proj = make_linear(self.hidden_size, self.num_heads * self.head_dim, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
        self.k_proj = make_linear(self.hidden_size, self.num_kv_heads * self.head_dim, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
        if not self.share_kv:
            self.v_proj = make_linear(self.hidden_size, self.num_kv_heads * self.head_dim, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
        else:
            self.v_proj = None
        self.o_proj = make_linear(self.num_heads * self.head_dim, self.hidden_size, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        self._setup_rope(config, attention_type)

    def inject_lora_projections(
        self,
        target_modules: list[str],
        adapter_factory: Callable[[nnx.Linear], LinearLayerProtocol],
    ) -> int:
        """Wrap matching attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) with LoRA adapters.

        Args:
            target_modules: List of target projection names to wrap.
            adapter_factory: Factory callable converting an `nnx.Linear` into a `LinearLayerProtocol` adapter.

        Returns:
            Number of projections wrapped in this attention block.

        """
        injected = 0
        if "q_proj" in target_modules and isinstance(self.q_proj, nnx.Linear):
            self.q_proj = adapter_factory(self.q_proj)
            injected += 1
        if "k_proj" in target_modules and isinstance(self.k_proj, nnx.Linear):
            self.k_proj = adapter_factory(self.k_proj)
            injected += 1
        if "v_proj" in target_modules and isinstance(self.v_proj, nnx.Linear):
            self.v_proj = adapter_factory(self.v_proj)
            injected += 1
        if "o_proj" in target_modules and isinstance(self.o_proj, nnx.Linear):
            self.o_proj = adapter_factory(self.o_proj)
            injected += 1
        return injected

    @jax.named_scope("gemma4_attention")
    def __call__(self, x: Array, positions: Array, cache: LayerCache | None = None, attention_mask: Array | None = None) -> Array:
        """Apply attention over the input sequences.

        Args:
            x: Input hidden state tensor of shape `(batch, seq_len, hidden_size)`.
            positions: Token position indices tensor of shape `(batch, seq_len)`.
            cache: Optional `LayerCache` for autoregressive KV caching.
            attention_mask: Optional additive attention mask tensor.

        Returns:
            Output hidden state tensor of shape `(batch, seq_len, hidden_size)`.

        """
        (batch_size, seq_len, _) = x.shape
        q = self.q_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        k = self.k_proj(x).reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = k if (self.share_kv or self.v_proj is None) else self.v_proj(x).reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        (q, k, v, mask, window) = _prepare_qkv_for_attention((q, k, v), positions, self.rope, cache)
        if _is_local_sliding_attention(self.attention_type):
            mask &= window < self.config.sliding_window_size
        structural_mask = jnp.where(mask, 0.0, MASK_PENALTY).astype(q.dtype)[:, None, :, :]
        attention_mask = structural_mask if attention_mask is None else attention_mask + structural_mask
        out = _compute_attention_scores_and_output((q, k, v), attention_mask, (self.head_dim, self.config.attn_logits_soft_cap, self.num_kv_heads, self.num_heads))
        if cache is not None:
            cache.cur_ind.value += seq_len
        res: Array = self.o_proj(out).astype(self.dtype)
        return res
