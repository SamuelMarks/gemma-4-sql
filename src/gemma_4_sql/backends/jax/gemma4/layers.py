"""Foundational Flax NNX layers (RMSNorm, SwiGLU MLP, ClippableLinear) for Gemma 4."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .config import ShardConfig


def make_linear(*args: Any, **kwargs: Any) -> Any:
    """Construct an `nnx.Linear` layer while stripping sharding metadata kwargs.

    Args:
        *args: Positional arguments forwarded to `nnx.Linear`.
        **kwargs: Keyword arguments forwarded to `nnx.Linear`.

    Returns:
        Configured `nnx.Linear` instance.

    """
    kwargs.pop("kernel_metadata", None)
    kwargs.pop("bias_metadata", None)
    return nnx.Linear(*args, **kwargs)


def make_embed(*args: Any, **kwargs: Any) -> Any:
    """Construct an `nnx.Embed` layer while stripping sharding metadata kwargs.

    Args:
        *args: Positional arguments forwarded to `nnx.Embed`.
        **kwargs: Keyword arguments forwarded to `nnx.Embed`.

    Returns:
        Configured `nnx.Embed` instance.

    """
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

    scale: nnx.Param[Array] | None

    def __init__(self, dim: int, eps: float = 1e-06, *, with_scale: bool = True, rngs: nnx.Rngs, **kwargs: object) -> None:
        """Initialize RMS normalization layer.

        Args:
            dim: Dimension of the input tensor.
            eps: Epsilon value for numerical stability.
            with_scale: Whether to include learnable scale parameter.
            rngs: Random number generators for weight initialization.
            **kwargs: Additional Flax/NNX module configuration parameters like 'dtype'.

        """
        self.eps = eps
        self.with_scale = with_scale
        self.dtype: Any = kwargs.get("dtype", jnp.float32)
        if self.with_scale:
            self.scale = nnx.Param(jax.nn.initializers.zeros(rngs.params(), (dim,), dtype=self.dtype))
        else:
            self.scale = None

    @jax.named_scope("gemma4_rms_norm")
    def __call__(self, x: Array) -> Array:
        """Apply RMS normalization.

        Args:
            x: Input array to normalize.

        Returns:
            RMS-normalized array cast to `self.dtype`.

        """
        xf32 = x.astype(jnp.float32)
        normed = xf32 * jax.lax.rsqrt(jnp.square(xf32).mean(-1, keepdims=True) + self.eps)
        if self.with_scale and self.scale is not None:
            scale_val = jnp.asarray(self.scale[...], dtype=jnp.float32)
            out = normed * (1.0 + scale_val)
        else:
            out = normed
        return out.astype(self.dtype)


class ConstVar(nnx.Variable[Any]):
    """Constant variable that should not be updated during training.

    This is used to store static tensors like inverse timescales for RoPE
    that need to be part of the model state but are not trainable parameters
    or mutable caches.
    """


class StatVar(nnx.Variable[Any]):
    """Statistical variable for tracking metrics like min/max values.

    This is used by layers like Gemma4ClippableLinear to track the bounds
    of activations for potential quantization or clipping purposes.
    """


class Gemma4ClippableLinear(nnx.Module):
    """Linear layer with optional input/output clipping."""

    def __init__(self, in_features: int, features: int, *, use_clipped_linears: bool = True, rngs: nnx.Rngs) -> None:
        """Initialize linear projection with optional activation range clipping.

        Args:
            in_features: Input feature dimension.
            features: Output feature dimension.
            use_clipped_linears: Whether to track and apply min/max clipping bounds.
            rngs: Flax NNX random number generators.

        """
        self.use_clipped_linears = use_clipped_linears
        self.linear = nnx.Linear(in_features, features, use_bias=False, rngs=rngs)
        if self.use_clipped_linears:
            self.input_min = StatVar(jnp.array(-jnp.inf))
            self.input_max = StatVar(jnp.array(jnp.inf))
            self.output_min = StatVar(jnp.array(-jnp.inf))
            self.output_max = StatVar(jnp.array(jnp.inf))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply a linear transformation, conditionally clipping the output.

        Args:
            x: Input feature tensor.

        Returns:
            Linearly projected (and optionally clipped) tensor.

        """
        if self.use_clipped_linears:
            x = jnp.clip(x, self.input_min[...], self.input_max[...])
        x = self.linear(x)
        if self.use_clipped_linears:
            x = jnp.clip(x, self.output_min[...], self.output_max[...])
        return x


class Gemma4MLP(nnx.Module):
    """Standard SwiGLU MLP used for both shared and routed experts."""

    def __init__(self, hidden_size: int, intermediate_size: int, *, rngs: nnx.Rngs, **kwargs: object) -> None:
        """Initialize SwiGLU MLP layer.

        Args:
            hidden_size: Hidden dimension of the model.
            intermediate_size: Intermediate projection dimension.
            rngs: Random number generators for weight initialization.
            **kwargs: Additional Flax/NNX module configuration parameters like 'dtype' and 'shd'.

        """
        raw_shd = kwargs.get("shd")
        shd = raw_shd if isinstance(raw_shd, ShardConfig) else ShardConfig.no_sharding()
        fc1_meta = {"out_sharding": shd.fc1_kernel} if shd.fc1_kernel is not None else {}
        fc1_bias_meta = {"out_sharding": shd.fc1_bias} if shd.fc1_bias is not None else {}
        fc2_meta = {"out_sharding": shd.fc2_kernel} if shd.fc2_kernel is not None else {}
        fc2_bias_meta = {"out_sharding": shd.fc2_bias} if shd.fc2_bias is not None else {}
        self.gate_proj = make_linear(hidden_size, intermediate_size, use_bias=False, kernel_metadata=fc1_meta, bias_metadata=fc1_bias_meta, rngs=rngs)
        self.up_proj = make_linear(hidden_size, intermediate_size, use_bias=False, kernel_metadata=fc1_meta, bias_metadata=fc1_bias_meta, rngs=rngs)
        self.down_proj = make_linear(intermediate_size, hidden_size, use_bias=False, kernel_metadata=fc2_meta, bias_metadata=fc2_bias_meta, rngs=rngs)
        self.dtype: Any = kwargs.get("dtype", jnp.float32)

    @jax.named_scope("gemma4_mlp")
    def __call__(self, x: Array, original_x: Array | None = None) -> Array:
        """Apply SwiGLU MLP transformation.

        Args:
            x: Input hidden state tensor.
            original_x: Optional unnormalized input tensor accepted for call-signature parity with `Gemma4MoE`.

        Returns:
            Projected hidden state tensor.

        """
        _ = original_x
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = jax.nn.silu(gate) * up
        out = self.down_proj(activated)
        return out.astype(self.dtype)
