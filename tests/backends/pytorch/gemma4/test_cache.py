"""Tests for cache modules."""

import torch

from gemma_4_sql.backends.pytorch.gemma4.cache import Cache, DynamicCache, StaticCache
from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config


def test_cache_base():
    """Test Cache base class."""
    cache = Cache()
    try:
        cache.update(torch.randn(1), torch.randn(1), 0)
    except NotImplementedError:
        pass
    try:
        cache.get_seq_length(0)
    except NotImplementedError:
        pass
    try:
        cache.get_max_length()
    except NotImplementedError:
        pass
    try:
        cache.reorder_cache(torch.randn(1))
    except NotImplementedError:
        pass


def test_dynamic_cache():
    """Test DynamicCache."""
    cache = DynamicCache()
    assert cache.get_max_length() is None
    assert cache.get_seq_length(0) == 0

    k = torch.randn(2, 4, 10, 64)
    v = torch.randn(2, 4, 10, 64)
    k_out, _v_out = cache.update(k, v, 0)
    assert cache.get_seq_length(0) == 10
    assert k_out.shape == (2, 4, 10, 64)

    k2 = torch.randn(2, 4, 5, 64)
    v2 = torch.randn(2, 4, 5, 64)
    k_out2, _v_out2 = cache.update(k2, v2, 0)
    assert cache.get_seq_length(0) == 15
    assert k_out2.shape == (2, 4, 15, 64)

    cache.reorder_cache(torch.tensor([1, 0]))
    assert cache.key_cache[0].shape == (2, 4, 15, 64)


def test_static_cache():
    """Test StaticCache."""
    config = Gemma4Config(num_hidden_layers=2, num_key_value_heads=2, head_dim=64)
    cache = StaticCache(config, max_batch_size=2, max_cache_len=20, device=torch.device("cpu"))

    assert cache.get_max_length() == 20
    assert cache.get_seq_length(0) == 0

    k = torch.randn(2, 2, 10, 64)
    v = torch.randn(2, 2, 10, 64)
    k_out, _v_out = cache.update(k, v, 0)
    assert k_out.shape == (2, 2, 10, 64)

    cache.seen_tokens = 10
    assert cache.get_seq_length(0) == 10

    cache.reorder_cache(torch.tensor([1, 0]))
    assert cache.key_cache[0].shape == (2, 2, 20, 64)
