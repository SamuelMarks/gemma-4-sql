"""Tests for cache modules with comprehensive branch coverage."""

from __future__ import annotations

import pytest
import torch

from gemma_4_sql.backends.pytorch.gemma4.cache import Cache, DynamicCache, StaticCache
from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config


class DummyConcreteCache(Cache):
    """Concrete subclass of Cache for testing interface compliance."""

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update concrete cache."""
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get sequence length."""
        return self.seen_tokens

    def get_max_length(self) -> int | None:
        """Get max length."""
        return 128

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """Reorder cache."""


def test_cache_abstract_instantiation() -> None:
    """Test that Cache cannot be directly instantiated."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class Cache"):
        Cache()  # type: ignore[abstract]

    concrete = DummyConcreteCache(seen_tokens=5, max_batch_size=4, device=torch.device("cpu"))
    assert concrete.seen_tokens == 5
    assert concrete.max_batch_size == 4
    assert concrete.get_max_length() == 128
    assert concrete.get_seq_length(0) == 5
    k = torch.randn(1, 1, 1, 1)
    k_res, v_res = concrete.update(k, k, 0)
    assert k_res.shape == k.shape
    assert v_res.shape == k.shape
    concrete.reorder_cache(torch.tensor([0]))


def test_dynamic_cache() -> None:
    """Test DynamicCache initialization, expansion, and reordering."""
    cache = DynamicCache()
    assert cache.get_max_length() is None
    assert cache.get_seq_length(0) == 0

    k = torch.randn(2, 4, 10, 64)
    v = torch.randn(2, 4, 10, 64)
    k_out, v_out = cache.update(k, v, 0)
    assert cache.get_seq_length(0) == 10
    assert k_out.shape == (2, 4, 10, 64)
    assert v_out.shape == (2, 4, 10, 64)

    k2 = torch.randn(2, 4, 5, 64)
    v2 = torch.randn(2, 4, 5, 64)
    k_out2, _v_out2 = cache.update(k2, v2, 0)
    assert cache.get_seq_length(0) == 15
    assert k_out2.shape == (2, 4, 15, 64)

    cache.reorder_cache(torch.tensor([1, 0]))
    assert cache.key_cache[0].shape == (2, 4, 15, 64)


def test_dynamic_cache_validation_errors() -> None:
    """Test DynamicCache validation raises ValueError on malformed inputs."""
    cache = DynamicCache()
    invalid_3d = torch.randn(2, 4, 10)
    valid_4d = torch.randn(2, 4, 10, 64)

    with pytest.raises(ValueError, match="key_states and value_states must be 4D tensors"):
        cache.update(invalid_3d, valid_4d, 0)

    with pytest.raises(ValueError, match="key_states and value_states must be 4D tensors"):
        cache.update(valid_4d, invalid_3d, 0)

    with pytest.raises(ValueError, match="layer_idx must be non-negative"):
        cache.update(valid_4d, valid_4d, -1)


def test_static_cache() -> None:
    """Test StaticCache allocation, updates, and sequence tracking."""
    config = Gemma4Config(num_hidden_layers=2, num_key_value_heads=2, head_dim=64)
    cache = StaticCache(config, max_batch_size=2, max_cache_len=20, device=torch.device("cpu"))

    assert cache.get_max_length() == 20
    assert cache.get_seq_length(0) == 0

    k = torch.randn(2, 2, 10, 64)
    v = torch.randn(2, 2, 10, 64)
    k_out, v_out = cache.update(k, v, 0)
    assert k_out.shape == (2, 2, 10, 64)
    assert v_out.shape == (2, 2, 10, 64)

    cache.seen_tokens = 10
    assert cache.get_seq_length(0) == 10

    cache.reorder_cache(torch.tensor([1, 0]))
    assert cache.key_cache[0].shape == (2, 2, 20, 64)


def test_static_cache_validation_errors() -> None:
    """Test StaticCache boundary checks and validation."""
    config = Gemma4Config(num_hidden_layers=1, num_key_value_heads=2, head_dim=64)
    cache = StaticCache(config, max_batch_size=2, max_cache_len=15, device=torch.device("cpu"))

    invalid_3d = torch.randn(2, 2, 5)
    valid_4d = torch.randn(2, 2, 5, 64)

    with pytest.raises(ValueError, match="key_states and value_states must be 4D tensors"):
        cache.update(invalid_3d, valid_4d, 0)

    with pytest.raises(ValueError, match=r"layer_idx .* out of range"):
        cache.update(valid_4d, valid_4d, 5)

    batch_too_large = torch.randn(4, 2, 5, 64)
    with pytest.raises(ValueError, match="exceeds StaticCache max_batch_size"):
        cache.update(batch_too_large, batch_too_large, 0)

    seq_too_large = torch.randn(2, 2, 20, 64)
    with pytest.raises(ValueError, match=r"Sequence length .* exceeds max_cache_len"):
        cache.update(seq_too_large, seq_too_large, 0)
