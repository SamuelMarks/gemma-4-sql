from unittest.mock import MagicMock

import jax.numpy as jnp

from gemma_4_sql.backends.jax.gemma4.cache import LayerCache, init_cache


def test_layer_cache():
    cache = LayerCache((2, 8, 4, 16), jnp.float32, None)
    assert cache.size == 8
    assert cache.cur_ind.value == 0
    assert cache.k_cache.value.shape == (2, 8, 4, 16)
    assert cache.v_cache.value.shape == (2, 8, 4, 16)


def test_init_cache():
    config = MagicMock()
    config.num_hidden_layers = 2
    # attention patterns have GLOBAL at index 5. so index 0 and 1 are LOCAL_SLIDING
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.dtype = jnp.float32
    config.shd_cfg.cache = None

    caches = init_cache(config, 2, 5)  # max_seq_len=5 -> cache_size=8 (next power of 2)
    assert len(caches) == 2
    assert caches[0].size == 8
    assert caches[0].k_cache.value.shape == (2, 8, 2, 16)


def test_init_cache_global():
    config = MagicMock()
    config.num_hidden_layers = 6  # 6th is global
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.num_global_key_value_heads = 4
    config.global_head_dim = 32
    config.dtype = jnp.float32
    config.shd_cfg.cache = None

    caches = init_cache(config, 2, 5)
    assert len(caches) == 6
    assert caches[5].k_cache.value.shape == (2, 8, 4, 32)


def test_init_cache_global_fallback():
    config = MagicMock()
    config.num_hidden_layers = 6  # 6th is global
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.num_global_key_value_heads = None
    config.global_head_dim = None
    config.dtype = jnp.float32
    config.shd_cfg.cache = None

    caches = init_cache(config, 2, 5)
    assert len(caches) == 6
    assert caches[5].k_cache.value.shape == (2, 8, 2, 16)
