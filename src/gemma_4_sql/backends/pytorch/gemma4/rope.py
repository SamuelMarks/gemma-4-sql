"""Rotary Positional Embeddings for Gemma 4."""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of `x`.

    Args:
        x: Input tensor whose last dimension is even.

    Returns:
        Tensor `[-x2, x1]` with rotated halves.

    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key projections.

    Args:
        q: Query projection tensor of shape `(batch, heads, seq_len, head_dim)`.
        k: Key projection tensor of shape `(batch, kv_heads, seq_len, head_dim)`.
        cos: Cached cosine table of shape `(max_seq_len, head_dim)`.
        sin: Cached sine table of shape `(max_seq_len, head_dim)`.
        position_ids: Token position index tensor of shape `(batch, seq_len)`.
        unsqueeze_dim: Head broadcast dimension index.

    Returns:
        Tuple of rotated `(q_embed, k_embed)` tensors.

    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Gemma4RotaryEmbedding(nn.Module):
    """Gemma 4 Rotary Embedding supporting linear scaling and proportional partial rotation."""

    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: torch.device | None = None,
        factor: float = 1.0,
        proportion: float = 1.0,
    ) -> None:
        """Initialize Gemma4RotaryEmbedding.

        Args:
            dim: Attention head dimension.
            max_position_embeddings: Initial maximum sequence length to cache.
            base: Rotary frequency base theta.
            device: Optional PyTorch device for frequency buffers.
            factor: Linear context scaling divisor applied to frequencies.
            proportion: Fraction `(0.0, 1.0]` of `dim` dimensions receiving rotary rotation;
                remaining dimensions receive zero frequency (`cos=1, sin=0`, pass-through).

        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.factor = factor
        self.proportion = proportion

        half_dim = self.dim // 2
        rotary_half_dim = max(0, min(half_dim, round(self.proportion * half_dim)))
        if rotary_half_dim == 0:
            inv_freq = torch.zeros((half_dim,), dtype=torch.float32, device=device)
        else:
            rotary_dim = rotary_half_dim * 2
            active_freq = 1.0 / (
                self.base
                ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64, device=device).float() / float(rotary_dim))
            )
            if self.factor != 1.0:
                active_freq = active_freq / float(self.factor)
            if rotary_half_dim < half_dim:
                pad_zeros = torch.zeros((half_dim - rotary_half_dim,), dtype=torch.float32, device=active_freq.device)
                inv_freq = torch.cat((active_freq, pad_zeros), dim=0)
            else:
                inv_freq = active_freq

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Populate cached cosine and sine rotary tables up to `seq_len`.

        Args:
            seq_len: Maximum token position length to precompute.
            device: Target PyTorch device.
            dtype: Target floating-point dtype.

        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(cos, sin)` rotary tables up to `seq_len`.

        Args:
            x: Reference tensor providing target `device` and `dtype`.
            seq_len: Required sequence length.

        Returns:
            Tuple of cached `(cos, sin)` embedding tensors of shape `(seq_len, dim)`.

        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class Gemma4RotaryEmbedding2D(nn.Module):
    """Gemma 4 2D Rotary Embedding for spatial vision patches."""

    inv_freq: torch.Tensor

    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0, device: torch.device | None = None
    ) -> None:
        """Initialize Gemma4RotaryEmbedding2D.

        Args:
            dim: Vision head dimension (split evenly across height and width axes).
            max_position_embeddings: Maximum spatial grid dimension.
            base: Rotary frequency base theta.
            device: Optional PyTorch device for frequency buffers.

        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.half_dim = dim // 2

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.half_dim, 2, dtype=torch.int64).float().to(device) / self.half_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D spatial `(cos, sin)` rotary embeddings for a `(height, width)` patch grid.

        Args:
            x: Reference activation tensor providing `device` and `dtype`.
            height: Grid height in patches.
            width: Grid width in patches.

        Returns:
            Tuple of `(cos, sin)` tensors of shape `(height * width, dim)`.

        """
        device = x.device
        dtype = x.dtype

        h_pos = torch.arange(height, device=device, dtype=torch.int64).type_as(self.inv_freq)
        w_pos = torch.arange(width, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs_h = torch.outer(h_pos, self.inv_freq)
        freqs_w = torch.outer(w_pos, self.inv_freq)

        freqs_h = freqs_h.unsqueeze(1).expand(-1, width, -1).reshape(-1, self.half_dim // 2)
        freqs_w = freqs_w.unsqueeze(0).expand(height, -1, -1).reshape(-1, self.half_dim // 2)

        freqs = torch.cat((freqs_h, freqs_w), dim=-1)
        emb = torch.cat((freqs, freqs), dim=-1)

        return emb.cos().to(dtype), emb.sin().to(dtype)
