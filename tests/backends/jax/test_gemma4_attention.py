"""Unit and behavioral tests for gemma4 attention."""

from unittest.mock import MagicMock

from tests.conftest import requires_jax

pytestmark = requires_jax

import jax.numpy as jnp
from flax import nnx

from gemma_4_sql.backends.jax.gemma4.attention import (
    Gemma4Attention,
    _compute_attention_scores_and_output,
    _prepare_qkv_for_attention,
)
from gemma_4_sql.backends.jax.gemma4.cache import LayerCache
from gemma_4_sql.backends.jax.gemma4.rope import RoPE


def test_compute_attention_scores_and_output():
    """Verify compute attention scores and output behavior."""
    # test soft_cap
    q = jnp.ones((2, 1, 4, 16))
    k = jnp.ones((2, 4, 4, 16))
    v = jnp.ones((2, 4, 4, 16))

    # head_dim=16, soft_cap=50.0, num_kv_heads=4, num_heads=4
    out = _compute_attention_scores_and_output((q, k, v), None, (16, 50.0, 4, 4))
    assert out.shape == (2, 1, 64)

    # test without soft_cap and with attention_mask
    mask = jnp.zeros((2, 4, 1, 4))
    out2 = _compute_attention_scores_and_output((q, k, v), mask, (16, None, 4, 4))
    assert out2.shape == (2, 1, 64)

    # test num_kv_heads != num_heads (handled inside _compute_attention_scores_and_output)


def test_prepare_qkv_for_attention():
    """Verify prepare qkv for attention behavior."""
    q = jnp.ones((2, 1, 4, 16))
    k = jnp.ones((2, 1, 4, 16))
    v = jnp.ones((2, 1, 4, 16))
    positions = jnp.array([[0], [1]])
    rope = RoPE(rope_type="default", head_dim=16, rope_theta=10000, factor=1.0)

    # without cache
    q_out, k_out, _v_out, mask, _window = _prepare_qkv_for_attention((q, k, v), positions, rope, None)
    assert q_out.shape == (2, 1, 4, 16)
    assert mask.shape == (2, 1, 1)

    # with cache
    cache = LayerCache((2, 8, 4, 16), jnp.float32)
    q_out, k_out, _v_out, mask, _window = _prepare_qkv_for_attention((q, k, v), positions, rope, cache)
    assert k_out.shape == (2, 8, 4, 16)


def test_gemma4_attention_local():
    """Verify gemma4 attention local behavior."""
    rngs = nnx.Rngs(0)
    config = MagicMock()
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.hidden_size = 64
    config.dtype = jnp.float32
    config.rms_norm_eps = 1e-6
    config.shd_cfg.norm = None
    config.local_rope_proportion = 1.0
    config.local_rope_max_timescale = 10000
    config.attn_logits_soft_cap = 50.0
    config.sliding_window_size = 512

    class LocalType:
        """Test stub for `LocalType`."""

        name = "LOCAL_SLIDING"

    from typing import Any

    loc_type: Any = LocalType()
    attn = Gemma4Attention(config, loc_type, rngs=rngs)

    x = jnp.ones((2, 3, 64))
    positions = jnp.array([[0, 1, 2], [0, 1, 2]])
    out = attn(x, positions)
    assert out.shape == (2, 3, 64)

    cache = LayerCache((2, 8, 2, 16), jnp.float32)
    out_cache = attn(x, positions, cache=cache)
    assert out_cache.shape == (2, 3, 64)
    assert cache.cur_ind.value == 3


def test_gemma4_attention_global():
    """Verify gemma4 attention global behavior."""
    rngs = nnx.Rngs(0)
    config = MagicMock()
    config.num_attention_heads = 4
    config.num_global_key_value_heads = 4
    config.global_head_dim = 16
    config.share_kv_projections = True
    config.hidden_size = 64
    config.dtype = jnp.float32
    config.rms_norm_eps = 1e-6
    config.shd_cfg.norm = None
    config.global_rope_proportion = 0.25
    config.global_rope_max_timescale = 100000
    config.attn_logits_soft_cap = 50.0

    class GlobalType:
        """Test stub for `GlobalType`."""

        name = "GLOBAL"

    from typing import Any

    glob_type: Any = GlobalType()
    attn = Gemma4Attention(config, glob_type, rngs=rngs)

    x = jnp.ones((2, 3, 64))
    positions = jnp.array([[0, 1, 2], [0, 1, 2]])
    mask = jnp.zeros((2, 1, 3, 3))
    out = attn(x, positions, attention_mask=mask)
    assert out.shape == (2, 3, 64)


def test_gemma4_attention_global_fallback_options():
    """Verify gemma4 attention global fallback options behavior."""
    from typing import Any

    rngs = nnx.Rngs(0)
    config = MagicMock()
    config.num_attention_heads = 4
    config.num_global_key_value_heads = None
    config.num_key_value_heads = 2
    config.global_head_dim = None
    config.head_dim = 16
    config.share_kv_projections = False
    config.hidden_size = 64
    config.dtype = jnp.float32
    config.rms_norm_eps = 1e-6
    config.shd_cfg.norm = None
    config.global_rope_proportion = 0.25
    # Let's test the getattr fallback for global_rope_max_timescale by removing it from the mock
    del config.global_rope_max_timescale
    config.rope_max_timescale = 10000

    class GlobalType:
        """Test stub for `GlobalType`."""

        name = "GLOBAL"

    glob_type: Any = GlobalType()
    attn = Gemma4Attention(config, glob_type, rngs=rngs)
    assert attn.num_kv_heads == 2
    assert attn.head_dim == 16
    assert attn.rope.rope_kwargs["rope_theta"] == 10000
