"""KV Cache definitions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import nnx

from .config import AttentionType

if TYPE_CHECKING:
    from jax.sharding import PartitionSpec

    from .config import ModelConfig


class LayerCache(nnx.Module):
    """KV Cache for a single decoder layer.

    Attributes
    ----------
        k_cache: The key cache tensor.
        v_cache: The value cache tensor.
        cur_ind: The current sequence index being written to.
        size: The maximum sequence length the cache can hold.

    """

    def __init__(self, cache_shape: tuple[int, int, int, int], dtype: jnp.dtype, _shd: PartitionSpec | None = None) -> None:
        """Docstring for __init__.

        Args:
            cache_shape: A sequence of cache shape.
            dtype: The dtype.
            _shd: The  shd.
        """
        self.k_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype))
        self.v_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype))
        self.cur_ind = nnx.Cache(jnp.zeros((), dtype=jnp.int32))
        self.size = cache_shape[1]


Cache = list[LayerCache]


Cache = list[LayerCache]


def init_cache(config: ModelConfig, batch_size: int, max_seq_len: int) -> Cache:
    """Initialize the KV cache for all layers.

    Args:
    ----
        config: The model configuration.
        batch_size: The batch size for generation.
        max_seq_len: The maximum sequence length to cache.

    Returns:
    -------
        A list of LayerCache objects, one for each hidden layer.

    """
    cache_size = 2 ** math.ceil(math.log2(max(max_seq_len, 1)))
    caches = []
    for i in range(config.num_hidden_layers):
        attn_type = GEMMA4_ATTENTION_PATTERN[i % len(GEMMA4_ATTENTION_PATTERN)]
        if attn_type == AttentionType.GLOBAL:
            num_kv = config.num_global_key_value_heads if config.num_global_key_value_heads is not None else config.num_key_value_heads
            hd = config.global_head_dim if config.global_head_dim is not None else config.head_dim
        else:
            num_kv = config.num_key_value_heads
            hd = config.head_dim
        caches.append(LayerCache((batch_size, cache_size, num_kv, hd), config.dtype, config.shd_cfg.cache))
    return caches


GEMMA4_ATTENTION_PATTERN = (AttentionType.LOCAL_SLIDING, AttentionType.LOCAL_SLIDING, AttentionType.LOCAL_SLIDING, AttentionType.LOCAL_SLIDING, AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL)


GEMMA4_ATTENTION_PATTERN = (AttentionType.LOCAL_SLIDING, AttentionType.LOCAL_SLIDING, AttentionType.LOCAL_SLIDING, AttentionType.LOCAL_SLIDING, AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL)
