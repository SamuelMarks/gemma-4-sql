"""Cache mechanisms for Gemma 4."""

from __future__ import annotations

import abc

import torch

from .config import Gemma4Config


class Cache(abc.ABC):
    """Abstract base class for KV-cache implementations in Gemma 4."""

    def __init__(
        self,
        seen_tokens: int = 0,
        max_batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize base Cache metadata tracking.

        Args:
            seen_tokens: Initial number of tokens previously processed.
            max_batch_size: Optional upper bound on supported batch size.
            device: Optional torch device where cached tensors are resident.

        """
        self.seen_tokens: int = seen_tokens
        self.max_batch_size: int | None = max_batch_size
        self.device: torch.device | None = device

    @abc.abstractmethod
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the cache and return the new key/value states.

        Args:
            key_states: Projected key tensor of shape `(batch, heads, seq_len, head_dim)`.
            value_states: Projected value tensor of shape `(batch, heads, seq_len, head_dim)`.
            layer_idx: Zero-based layer index within the model transformer.

        Returns:
            Tuple of updated key states and value states for attention computation.

        Raises:
            ValueError: If input tensor dimensions or layer index are invalid.

        """

    @abc.abstractmethod
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Provide the current sequence length of the cache.

        Args:
            layer_idx: Zero-based layer index to query.

        Returns:
            Current cached sequence length for the specified layer.

        """

    @abc.abstractmethod
    def get_max_length(self) -> int | None:
        """Provide the maximum length the cache can hold.

        Returns:
            Maximum sequence length integer, or `None` if dynamically bounded.

        """

    @abc.abstractmethod
    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """Reorder the cache for beam search.

        Args:
            beam_idx: 1D tensor of beam indices to gather.

        """


class DynamicCache(Cache):
    """Dynamic cache for autoregressive generation that grows dynamically."""

    def __init__(self) -> None:
        """Initialize DynamicCache with empty key and value caches."""
        super().__init__(seen_tokens=0, max_batch_size=None, device=None)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the cache with new key and value states.

        Args:
            key_states: Projected key tensor of shape `(batch, heads, seq_len, head_dim)`.
            value_states: Projected value tensor of shape `(batch, heads, seq_len, head_dim)`.
            layer_idx: Zero-based layer index within the model transformer.

        Returns:
            Tuple of updated key states and value states up to current length.

        Raises:
            ValueError: If tensor dimensions are not 4D or `layer_idx` is negative.

        """
        if key_states.dim() != 4 or value_states.dim() != 4:
            msg = f"key_states and value_states must be 4D tensors, got {key_states.shape} and {value_states.shape}."
            raise ValueError(msg)
        if layer_idx < 0:
            msg = f"layer_idx must be non-negative, got {layer_idx}."
            raise ValueError(msg)

        if self.device is None:
            self.device = key_states.device

        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.empty(0, dtype=key_states.dtype, device=key_states.device))
            self.value_cache.append(torch.empty(0, dtype=value_states.dtype, device=value_states.device))

        if self.key_cache[layer_idx].numel() == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        new_len = int(self.key_cache[layer_idx].shape[2])
        if layer_idx == 0 or new_len > self.seen_tokens:
            self.seen_tokens = new_len
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Provide the sequence length of the specified layer.

        Args:
            layer_idx: Zero-based layer index to query.

        Returns:
            Sequence length integer.

        """
        if layer_idx < 0 or len(self.key_cache) <= layer_idx or self.key_cache[layer_idx].numel() == 0:
            return 0
        return int(self.key_cache[layer_idx].shape[2])

    def get_max_length(self) -> int | None:
        """Return the maximum length (`None` for dynamic cache).

        Returns:
            Always `None` as `DynamicCache` expands dynamically with sequence growth.

        """
        return None

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """Reorder the cache for beam search across all layers.

        Args:
            beam_idx: 1D integer tensor containing beam indices to select.

        """
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel() > 0:
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)


class StaticCache(Cache):
    """Static cache pre-allocated per layer for `torch.compile` and CUDA graphs."""

    def __init__(
        self,
        config: Gemma4Config,
        max_batch_size: int,
        max_cache_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize StaticCache with per-layer local/global pre-allocated tensors.

        Args:
            config: Model configuration specifying local/global head dimensions and KV head counts.
            max_batch_size: Pre-allocated maximum batch size.
            max_cache_len: Pre-allocated maximum sequence length.
            device: Target torch compute device.
            dtype: Floating point precision for cached activations.

        """
        super().__init__(seen_tokens=0, max_batch_size=max_batch_size, device=device)
        self.max_cache_len: int = max_cache_len
        self.head_dim: int = int(config.head_dim)
        self.num_key_value_heads: int = int(config.num_key_value_heads)
        self._layer_seen_tokens: list[int] = [0] * config.num_hidden_layers

        global_layers = set(config.global_attn_layers)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        for layer_idx in range(config.num_hidden_layers):
            is_global = bool(layer_idx in global_layers or config.is_global)
            if is_global:
                kv_heads = (
                    int(config.num_global_key_value_heads)
                    if config.num_global_key_value_heads is not None
                    else self.num_key_value_heads
                )
                layer_hd = int(config.global_head_dim) if config.global_head_dim is not None else self.head_dim
            else:
                kv_heads = self.num_key_value_heads
                layer_hd = self.head_dim
            self.key_cache.append(
                torch.zeros(
                    (max_batch_size, kv_heads, max_cache_len, layer_hd),
                    dtype=dtype,
                    device=device,
                )
            )
            self.value_cache.append(
                torch.zeros(
                    (max_batch_size, kv_heads, max_cache_len, layer_hd),
                    dtype=dtype,
                    device=device,
                )
            )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new key/value states at the layer's write offset and advance sequence length.

        Args:
            key_states: Projected key tensor of shape `(batch, heads, seq_len, head_dim)`.
            value_states: Projected value tensor of shape `(batch, heads, seq_len, head_dim)`.
            layer_idx: Zero-based layer index within the model transformer.

        Returns:
            Tuple of cached key states and value states sliced up to the updated length.

        Raises:
            ValueError: If tensor rank, `layer_idx`, `batch_size`, or sequence length is invalid.

        """
        if key_states.dim() != 4 or value_states.dim() != 4:
            msg = f"key_states and value_states must be 4D tensors, got {key_states.shape} and {value_states.shape}."
            raise ValueError(msg)
        if layer_idx < 0 or layer_idx >= len(self.key_cache):
            msg = f"layer_idx {layer_idx} out of range [0, {len(self.key_cache)})."
            raise ValueError(msg)

        batch_size, _, seq_len, _ = key_states.shape
        if self.max_batch_size is not None and batch_size > self.max_batch_size:
            msg = f"batch_size {batch_size} exceeds StaticCache max_batch_size {self.max_batch_size}."
            raise ValueError(msg)

        offset = self._layer_seen_tokens[layer_idx]
        new_len = offset + seq_len
        if new_len > self.max_cache_len:
            msg = f"Sequence length {new_len} exceeds max_cache_len {self.max_cache_len}."
            raise ValueError(msg)

        self.key_cache[layer_idx][:batch_size, :, offset:new_len, :] = key_states
        self.value_cache[layer_idx][:batch_size, :, offset:new_len, :] = value_states
        self._layer_seen_tokens[layer_idx] = new_len
        if layer_idx == 0 or new_len > self.seen_tokens:
            self.seen_tokens = new_len

        return (
            self.key_cache[layer_idx][:batch_size, :, :new_len, :],
            self.value_cache[layer_idx][:batch_size, :, :new_len, :],
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Provide the current cached sequence length for `layer_idx`.

        Args:
            layer_idx: Zero-based layer index to query.

        Returns:
            Current cached sequence length for `layer_idx` (`0` if out of range).

        """
        if 0 <= layer_idx < len(self._layer_seen_tokens):
            return self._layer_seen_tokens[layer_idx]
        return 0

    def get_max_length(self) -> int | None:
        """Provide the maximum sequence length the static cache can hold.

        Returns:
            Maximum sequence length integer.

        """
        return self.max_cache_len

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """Reorder the cache for beam search across all layers.

        Args:
            beam_idx: 1D integer tensor containing beam indices to select.

        """
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)
