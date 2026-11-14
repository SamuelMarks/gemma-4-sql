"""Unit and behavioral tests for gemma4 cache."""

from __future__ import annotations

from tests.conftest import requires_jax

pytestmark = requires_jax

from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.jax.gemma4.cache import LayerCache, init_cache
from gemma_4_sql.backends.jax.gemma4.config import ModelConfig


def test_layer_cache() -> None:
    """Verify layer cache behavior."""
    cache = LayerCache((2, 8, 4, 16), jnp.float32, None)
    assert cache.size == 8
    assert cache.cur_ind.value == 0
    assert cache.k_cache.value.shape == (2, 8, 4, 16)
    assert cache.v_cache.value.shape == (2, 8, 4, 16)


def test_init_cache() -> None:
    """Verify init cache behavior."""
    config = ModelConfig(
        num_hidden_layers=2,
        num_key_value_heads=2,
        head_dim=16,
        dtype=jnp.float32,
    )

    caches = init_cache(config, 2, 5)
    assert len(caches) == 2
    assert caches[0].size == 8
    assert caches[0].k_cache.value.shape == (2, 8, 2, 16)


def test_init_cache_global() -> None:
    """Verify init cache global behavior."""
    config = ModelConfig(
        num_hidden_layers=6,
        num_key_value_heads=2,
        head_dim=16,
        num_global_key_value_heads=4,
        global_head_dim=32,
        dtype=jnp.float32,
    )

    caches = init_cache(config, 2, 5)
    assert len(caches) == 6
    assert caches[5].k_cache.value.shape == (2, 8, 4, 32)


def test_init_cache_global_fallback() -> None:
    """Verify init cache global fallback behavior."""
    config = ModelConfig(
        num_hidden_layers=6,
        num_key_value_heads=2,
        head_dim=16,
        num_global_key_value_heads=None,
        global_head_dim=None,
        dtype=jnp.float32,
    )

    caches = init_cache(config, 2, 5)
    assert len(caches) == 6
    assert caches[5].k_cache.value.shape == (2, 8, 2, 16)
