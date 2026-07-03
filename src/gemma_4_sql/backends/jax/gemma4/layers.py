"""Provide module docstring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .config import ShardConfig

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONValue


def _make_linear(*args: object, **kwargs: JSONValue) -> object:
    """Docstring for _make_linear."""
    kwargs.pop("kernel_metadata", None)
    kwargs.pop("bias_metadata", None)
    return nnx.Linear(*args, **kwargs)


def _make_embed(*args: object, **kwargs: JSONValue) -> object:
    """Docstring for _make_embed."""
    kwargs.pop("embedding_metadata", None)
    return nnx.Embed(*args, **kwargs)


class Gemma4RMSNorm(nnx.Module):
    """RMSNorm layer for Gemma 4.

    Gemma 4 models typically use an offset scale (`1.0 + scale`) for normal layers,
    but MoE gate norms and v_norm require `with_scale=False` (no learned scale).

    Attributes
    ----------
        dim: The input dimension.
        eps: Epsilon to prevent division by zero.
        with_scale: Whether to include a learned scale parameter.
        dtype: The data type for computation.

    """

    def __init__(self, dim: int, eps: float = 1e-06, *, with_scale: bool = True, rngs: nnx.Rngs, **kwargs: object) -> None:
        """Docstring for __init__."""
        self.eps = eps
        self.with_scale = with_scale
        self.dtype = kwargs.get("dtype", jnp.float32)
        if self.with_scale:
            self.scale = nnx.Param(jax.nn.initializers.zeros(rngs.params(), dim, dtype=self.dtype))
        else:
            self.scale = None

    @jax.named_scope("gemma4_rms_norm")
    def __call__(self, x: Array) -> Array:
        """Apply RMS normalization."""
        xf32 = x.astype(jnp.float32)
        normed = xf32 * jax.lax.rsqrt(jnp.square(xf32).mean(-1, keepdims=True) + self.eps)
        if self.with_scale:
            scale_val = jnp.asarray(self.scale[...], dtype=jnp.float32)
            out = normed * (1.0 + scale_val)
        else:
            out = normed
        return out.astype(self.dtype)


class ConstVar(nnx.Variable):
    """Constant variable that should not be updated during training.

    This is used to store static tensors like inverse timescales for RoPE
    that need to be part of the model state but are not trainable parameters
    or mutable caches.
    """


class StatVar(nnx.Variable):
    """Statistical variable for tracking metrics like min/max values.

    This is used by layers like Gemma4ClippableLinear to track the bounds
    of activations for potential quantization or clipping purposes.
    """


class Gemma4ClippableLinear(nnx.Module):
    """Linear layer with optional input/output clipping."""

    def __init__(self, in_features: int, features: int, *, use_clipped_linears: bool = True, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.use_clipped_linears = use_clipped_linears
        self.linear = nnx.Linear(in_features, features, use_bias=False, rngs=rngs)
        if self.use_clipped_linears:
            self.input_min = StatVar(jnp.array(-jnp.inf))
            self.input_max = StatVar(jnp.array(jnp.inf))
            self.output_min = StatVar(jnp.array(-jnp.inf))
            self.output_max = StatVar(jnp.array(jnp.inf))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply a linear transformation, conditionally clipping the output."""
        if self.use_clipped_linears:
            x = jnp.clip(x, self.input_min[...], self.input_max[...])
        x = self.linear(x)
        if self.use_clipped_linears:
            x = jnp.clip(x, self.output_min[...], self.output_max[...])
        return x


class Gemma4MLP(nnx.Module):
    """Standard SwiGLU MLP used for both shared and routed experts."""

    def __init__(self, hidden_size: int, intermediate_size: int, *, rngs: nnx.Rngs, **kwargs: object) -> None:
        """Docstring for __init__."""
        shd = kwargs.get("shd")
        if shd is None:
            shd = ShardConfig.no_sharding()
        self.gate_proj = _make_linear(hidden_size, intermediate_size, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
        self.up_proj = _make_linear(hidden_size, intermediate_size, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
        self.down_proj = _make_linear(intermediate_size, hidden_size, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
        self.dtype = kwargs.get("dtype", jnp.float32)

    @jax.named_scope("gemma4_mlp")
    def __call__(self, x: Array) -> Array:
        """Apply SwiGLU MLP transformation."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = jax.nn.silu(gate) * up
        out = self.down_proj(activated)
        return out.astype(self.dtype)
