"""Tests for cache modules with comprehensive behavioral and branch coverage."""

from __future__ import annotations

import pytest
import torch

from gemma_4_sql.backends.pytorch.gemma4.cache import Cache, DynamicCache, StaticCache
from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config


class DummyConcreteCache(Cache):
    """Concrete Cache subclass used to verify abstract base class interface behavior."""

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache states.

        Args:
            key_states: Key projection tensor.
            value_states: Value projection tensor.
            layer_idx: Zero-based layer index.

        Returns:
            Unmodified `(key_states, value_states)` tuple.

        """
        _ = layer_idx
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return cached sequence length.

        Args:
            layer_idx: Zero-based layer index.

        Returns:
            `seen_tokens` value.

        """
        _ = layer_idx
        return self.seen_tokens

    def get_max_length(self) -> int | None:
        """Return maximum cache capacity.

        Returns:
            Constant `128`.

        """
        return 128

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """No-op beam reordering for base contract test.

        Args:
            beam_idx: 1D beam index tensor.

        """
        _ = beam_idx


def test_base_cache_contract() -> None:
    """Verify abstract `Cache` base class initializes metadata and dispatches concrete methods."""
    dummy = DummyConcreteCache(seen_tokens=3, max_batch_size=4, device=torch.device("cpu"))
    assert dummy.seen_tokens == 3
    assert dummy.max_batch_size == 4
    assert dummy.device == torch.device("cpu")
    k = torch.ones((2, 2, 3, 8))
    v = torch.zeros((2, 2, 3, 8))
    k_out, v_out = dummy.update(k, v, 0)
    assert torch.equal(k_out, k)
    assert torch.equal(v_out, v)
    assert dummy.get_seq_length(0) == 3
    assert dummy.get_max_length() == 128
    dummy.reorder_cache(torch.tensor([1, 0]))


def test_dynamic_cache() -> None:
    """Verify `DynamicCache` appends tokens across multiple layers, reorders beams, and validates inputs."""
    cache = DynamicCache()
    assert cache.get_seq_length(0) == 0
    assert cache.get_seq_length(-1) == 0
    assert cache.get_max_length() is None

    k1 = torch.randn(2, 2, 4, 16)
    v1 = torch.randn(2, 2, 4, 16)
    k_out1, v_out1 = cache.update(k1, v1, 0)
    assert torch.equal(k_out1, k1)
    assert torch.equal(v_out1, v1)
    assert cache.get_seq_length(0) == 4
    assert cache.seen_tokens == 4

    # Sparse layer index > 0 allocates intermediate empty layer slots
    k_layer2 = torch.randn(2, 2, 4, 16)
    v_layer2 = torch.randn(2, 2, 4, 16)
    cache.update(k_layer2, v_layer2, 2)
    assert cache.get_seq_length(1) == 0
    assert cache.get_seq_length(2) == 4

    # Append 3 new tokens to layer 0
    k2 = torch.randn(2, 2, 3, 16)
    v2 = torch.randn(2, 2, 3, 16)
    k_out2, v_out2 = cache.update(k2, v2, 0)
    assert k_out2.shape == (2, 2, 7, 16)
    assert torch.equal(k_out2[:, :, :4, :], k1)
    assert torch.equal(k_out2[:, :, 4:7, :], k2)
    assert torch.equal(v_out2[:, :, 4:7, :], v2)
    assert cache.get_seq_length(0) == 7
    assert cache.seen_tokens == 7

    # Reorder beams while layer 1 remains empty (`numel() == 0` branch)
    cache.reorder_cache(torch.tensor([1, 0], dtype=torch.long))
    assert torch.equal(cache.key_cache[0][0, :, :4, :], k1[1])

    with pytest.raises(ValueError, match="must be 4D tensors"):
        cache.update(torch.randn(2, 4, 16), v2, 0)

    with pytest.raises(ValueError, match="must be non-negative"):
        cache.update(k2, v2, -1)


def test_static_cache_multi_step_and_heterogeneous_global_local_dims() -> None:
    """Verify `StaticCache.update` advances per-layer `seen_tokens` and supports global/local head dims."""
    config = Gemma4Config(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        global_attn_layers=[1],
        enable_multimodal=False,
    )
    cache = StaticCache(config, max_batch_size=2, max_cache_len=16, device=torch.device("cpu"))
    assert cache.get_max_length() == 16
    assert cache.get_seq_length(0) == 0
    assert cache.get_seq_length(1) == 0
    assert cache.get_seq_length(-1) == 0
    assert cache.get_seq_length(99) == 0
    assert cache.key_cache[0].shape == (2, 2, 16, 16)
    assert cache.key_cache[1].shape == (2, 1, 16, 32)

    # Step 1: prefill 10 tokens across local layer 0 and global layer 1 WITHOUT manually touching seen_tokens
    k0_step1 = torch.randn(2, 2, 10, 16)
    v0_step1 = torch.randn(2, 2, 10, 16)
    k0_out1, v0_out1 = cache.update(k0_step1, v0_step1, 0)
    assert k0_out1.shape == (2, 2, 10, 16)
    assert torch.equal(k0_out1, k0_step1)
    assert torch.equal(v0_out1, v0_step1)
    assert cache.get_seq_length(0) == 10
    assert cache.get_seq_length(1) == 0
    assert cache.seen_tokens == 10

    k1_step1 = torch.randn(2, 1, 10, 32)
    v1_step1 = torch.randn(2, 1, 10, 32)
    k1_out1, v1_out1 = cache.update(k1_step1, v1_step1, 1)
    assert k1_out1.shape == (2, 1, 10, 32)
    assert torch.equal(k1_out1, k1_step1)
    assert torch.equal(v1_out1, v1_step1)
    assert cache.get_seq_length(1) == 10

    # Step 2: decode 5 additional tokens and verify step 1 prefix is preserved intact and step 2 is appended at [10:15]
    k0_step2 = torch.randn(2, 2, 5, 16)
    v0_step2 = torch.randn(2, 2, 5, 16)
    k0_out2, v0_out2 = cache.update(k0_step2, v0_step2, 0)
    assert k0_out2.shape == (2, 2, 15, 16)
    assert cache.get_seq_length(0) == 15
    assert cache.seen_tokens == 15
    assert torch.equal(k0_out2[:, :, :10, :], k0_step1)
    assert torch.equal(k0_out2[:, :, 10:15, :], k0_step2)
    assert torch.equal(v0_out2[:, :, :10, :], v0_step1)
    assert torch.equal(v0_out2[:, :, 10:15, :], v0_step2)

    # Beam reordering swaps batch rows across all pre-allocated buffers
    cache.reorder_cache(torch.tensor([1, 0], dtype=torch.long))
    assert torch.equal(cache.key_cache[0][0, :, :10, :], k0_step1[1])
    assert torch.equal(cache.key_cache[0][0, :, 10:15, :], k0_step2[1])

    # Validation error branches
    with pytest.raises(ValueError, match="must be 4D tensors"):
        cache.update(torch.randn(2, 2, 16), v0_step2, 0)

    with pytest.raises(ValueError, match="out of range"):
        cache.update(k0_step2, v0_step2, 5)

    with pytest.raises(ValueError, match="exceeds StaticCache max_batch_size"):
        cache.update(torch.randn(3, 2, 1, 16), torch.randn(3, 2, 1, 16), 0)

    with pytest.raises(ValueError, match="exceeds max_cache_len"):
        cache.update(torch.randn(2, 2, 2, 16), torch.randn(2, 2, 2, 16), 0)
