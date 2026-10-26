"""Layers for Gemma 4."""

from __future__ import annotations

import torch
from torch import nn

from .config import Gemma4Config


class Gemma4RMSNorm(nn.Module):
    """RMSNorm for Gemma 4 supporting optional learned scale parameter."""

    weight: nn.Parameter | None

    def __init__(self, dim: int, eps: float = 1e-6, *, with_scale: bool = True, use_scale: bool = True) -> None:
        """Initialize Gemma4RMSNorm.

        Args:
            dim: Normalization feature dimension.
            eps: Numerical stability epsilon added to variance.
            with_scale: Whether to include a learned multiplicative scale parameter.
            use_scale: Alias for `with_scale` for cross-backend compatibility.

        """
        super().__init__()
        self.eps = eps
        self.with_scale = bool(with_scale and use_scale)
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply root-mean-square normalization without learned scaling.

        Args:
            x: Input floating-point tensor.

        Returns:
            Normalized tensor of the same shape as `x`.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for RMSNorm.

        Args:
            x: Input tensor to normalize.

        Returns:
            Normalized and optionally scaled tensor.

        """
        output = self._norm(x.float()).type_as(x)
        if self.with_scale and self.weight is not None:
            return output * self.weight
        return output


class Gemma4MLP(nn.Module):
    """Gated MLP for Gemma 4."""

    def __init__(self, config: Gemma4Config, intermediate_size: int | None = None) -> None:
        """Initialize Gemma4MLP.

        Args:
            config: Model configuration parameters.
            intermediate_size: Optional override for the MLP intermediate dimension.

        """
        super().__init__()
        inter_dim = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inter_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, config.hidden_size, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for MLP.

        Args:
            x: Input hidden state tensor.

        Returns:
            Projected output tensor.

        """
        res: torch.Tensor = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return res
