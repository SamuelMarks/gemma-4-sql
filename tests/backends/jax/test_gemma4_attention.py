"""Unit and behavioral tests for gemma4 attention."""

from __future__ import annotations

from tests.conftest import requires_jax

pytestmark = requires_jax

import jax.numpy as jnp
from flax import nnx

from gemma_4_sql.backends.jax.gemma4.attention import (
    Gemma4Attention,
    _compute_attention_scores_and_output,
    _is_global_attention,
    _is_local_sliding_attention,
    _prepare_qkv_for_attention,
)
from gemma_4_sql.backends.jax.gemma4.cache import LayerCache
from gemma_4_sql.backends.jax.gemma4.config import AttentionType, ModelConfig
from gemma_4_sql.backends.jax.gemma4.rope import RoPE


def test_compute_attention_scores_and_output() -> None:
    """Verify compute attention scores and output behavior."""
    q = jnp.ones((2, 1, 4, 16))
    k = jnp.ones((2, 4, 4, 16))
    v = jnp.ones((2, 4, 4, 16))

    out = _compute_attention_scores_and_output((q, k, v), None, (16, 50.0, 4, 4))
    assert out.shape == (2, 1, 64)

    mask = jnp.zeros((2, 4, 1, 4))
    out2 = _compute_attention_scores_and_output((q, k, v), mask, (16, None, 4, 4))
    assert out2.shape == (2, 1, 64)


def test_prepare_qkv_for_attention() -> None:
    """Verify prepare qkv for attention behavior."""
    q = jnp.ones((2, 1, 4, 16))
    k = jnp.ones((2, 1, 4, 16))
    v = jnp.ones((2, 1, 4, 16))
    positions = jnp.array([[0], [1]])
    rope = RoPE(rope_type="default", head_dim=16, rope_theta=10000, factor=1.0)

    q_out, k_out, _v_out, mask, _window = _prepare_qkv_for_attention((q, k, v), positions, rope, None)
    assert q_out.shape == (2, 1, 4, 16)
    assert mask.shape == (2, 1, 1)

    cache = LayerCache((2, 8, 4, 16), jnp.float32)
    q_out, k_out, _v_out, mask, _window = _prepare_qkv_for_attention((q, k, v), positions, rope, cache)
    assert k_out.shape == (2, 8, 4, 16)


def test_attention_type_helpers() -> None:
    """Verify _is_global_attention and _is_local_sliding_attention accept enum and string forms."""
    assert _is_global_attention(AttentionType.GLOBAL) is True
    assert _is_global_attention(AttentionType.LOCAL_SLIDING) is False
    assert _is_global_attention("GLOBAL") is True
    assert _is_global_attention("LOCAL_SLIDING") is False

    assert _is_local_sliding_attention(AttentionType.LOCAL_SLIDING) is True
    assert _is_local_sliding_attention(AttentionType.GLOBAL) is False
    assert _is_local_sliding_attention("LOCAL_SLIDING") is True
    assert _is_local_sliding_attention("GLOBAL") is False


def test_gemma4_attention_local() -> None:
    """Verify gemma4 attention local behavior."""
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_size=64,
        dtype=jnp.float32,
        rms_norm_eps=1e-6,
        local_rope_proportion=1.0,
        local_rope_max_timescale=10000,
        attn_logits_soft_cap=50.0,
        sliding_window_size=512,
    )
    attn = Gemma4Attention(config, AttentionType.LOCAL_SLIDING, rngs=rngs)

    x = jnp.ones((2, 3, 64))
    positions = jnp.array([[0, 1, 2], [0, 1, 2]])
    out = attn(x, positions)
    assert out.shape == (2, 3, 64)

    cache = LayerCache((2, 8, 2, 16), jnp.float32)
    out_cache = attn(x, positions, cache=cache)
    assert out_cache.shape == (2, 3, 64)
    assert cache.cur_ind.value == 3


def test_gemma4_attention_global() -> None:
    """Verify gemma4 attention global behavior."""
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        num_attention_heads=4,
        num_global_key_value_heads=4,
        global_head_dim=16,
        share_kv_projections=True,
        hidden_size=64,
        dtype=jnp.float32,
        rms_norm_eps=1e-6,
        global_rope_proportion=0.25,
        global_rope_max_timescale=100000,
        attn_logits_soft_cap=50.0,
    )
    attn = Gemma4Attention(config, AttentionType.GLOBAL, rngs=rngs)

    x = jnp.ones((2, 3, 64))
    positions = jnp.array([[0, 1, 2], [0, 1, 2]])
    mask = jnp.zeros((2, 1, 3, 3))
    out = attn(x, positions, attention_mask=mask)
    assert out.shape == (2, 3, 64)


def test_gemma4_attention_global_fallback_options() -> None:
    """Verify gemma4 attention global fallback options behavior."""
    rngs = nnx.Rngs(0)
    config = ModelConfig(
        num_attention_heads=4,
        num_global_key_value_heads=None,
        num_key_value_heads=2,
        global_head_dim=None,
        head_dim=16,
        share_kv_projections=False,
        hidden_size=64,
        dtype=jnp.float32,
        rms_norm_eps=1e-6,
        global_rope_proportion=0.25,
        global_rope_max_timescale=None,
        rope_max_timescale=10000,
    )
    attn = Gemma4Attention(config, AttentionType.GLOBAL, rngs=rngs)
    assert attn.num_kv_heads == 2
    assert attn.head_dim == 16
    assert attn.rope.rope_kwargs["rope_theta"] == 10000
