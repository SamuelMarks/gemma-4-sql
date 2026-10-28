"""Foundational Flax NNX layers (RMSNorm, SwiGLU MLP, ClippableLinear) for Gemma 4."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.sharding import PartitionSpec

from .config import JaxDTypeLike, ShardConfig


@runtime_checkable
class LinearLayerProtocol(Protocol):
    """Structural protocol for linear projection layers (`nnx.Linear` and `LoRALinearAdapter`)."""

    @property
    def kernel(self) -> nnx.Param[Array]:
        """Provide the projection weight kernel parameter."""
        ...

    @property
    def bias(self) -> nnx.Param[Array] | None:
        """Provide the optional additive bias parameter."""
        ...

    def __call__(self, x: Array, /) -> Array:
        """Apply linear projection to `x`."""
        ...


def make_linear(
    in_features: int,
    out_features: int,
    *,
    use_bias: bool = True,
    dtype: JaxDTypeLike | None = None,
    kernel_metadata: Mapping[str, PartitionSpec | str | None] | None = None,
    bias_metadata: Mapping[str, PartitionSpec | str | None] | None = None,
    rngs: nnx.Rngs,
) -> nnx.Linear:
    """Construct an `nnx.Linear` layer while stripping sharding metadata kwargs.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        use_bias: Whether to include an additive bias parameter.
        dtype: Optional computation dtype.
        kernel_metadata: Optional sharding metadata for kernel parameter.
        bias_metadata: Optional sharding metadata for bias parameter.
        rngs: Flax NNX random number generators.

    Returns:
        Configured `nnx.Linear` instance.

    """
    _ = (kernel_metadata, bias_metadata)
    return nnx.Linear(in_features, out_features, use_bias=use_bias, dtype=dtype, rngs=rngs)


def make_embed(
    num_embeddings: int,
    features: int,
    *,
    dtype: JaxDTypeLike | None = None,
    embedding_metadata: Mapping[str, PartitionSpec | str | None] | None = None,
    rngs: nnx.Rngs,
) -> nnx.Embed:
    """Construct an `nnx.Embed` layer while stripping sharding metadata kwargs.

    Args:
        num_embeddings: Vocabulary size or number of embeddings.
        features: Embedding vector dimension.
        dtype: Optional computation dtype.
        embedding_metadata: Optional sharding metadata for embedding table.
        rngs: Flax NNX random number generators.

    Returns:
        Configured `nnx.Embed` instance.

    """
    _ = embedding_metadata
    return nnx.Embed(num_embeddings, features, dtype=dtype, rngs=rngs)


class Gemma4RMSNorm(nnx.Module):
    """RMSNorm layer for Gemma 4.

    Gemma 4 models typically use an offset scale (`1.0 + scale`) for normal layers,
    but MoE gate norms and v_norm require `with_scale=False` (no learned scale).

    Attributes:
        dim: The input dimension.
        eps: Epsilon to prevent division by zero.
        with_scale: Whether to include a learned scale parameter.
        dtype: The data type for computation.

    """

    scale: nnx.Param[Array] | None

    def __init__(
        self,
        dim: int,
        eps: float = 1e-06,
        *,
        with_scale: bool = True,
        dtype: JaxDTypeLike = jnp.float32,
        _shd: PartitionSpec | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize RMS normalization layer.

        Args:
            dim: Dimension of the input tensor.
            eps: Epsilon value for numerical stability.
            with_scale: Whether to include learnable scale parameter.
            dtype: Data type for normalized outputs and scale parameter.
            _shd: Optional sharding specification.
            rngs: Random number generators for weight initialization.

        """
        _ = _shd
        self.eps = eps
        self.with_scale = with_scale
        self.dtype: JaxDTypeLike = dtype
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


class ConstVar(nnx.Variable[Array]):
    """Constant variable that should not be updated during training.

    This is used to store static tensors like inverse timescales for RoPE
    that need to be part of the model state but are not trainable parameters
    or mutable caches.
    """


class StatVar(nnx.Variable[Array]):
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

    gate_proj: LinearLayerProtocol
    up_proj: LinearLayerProtocol
    down_proj: LinearLayerProtocol

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        rngs: nnx.Rngs,
        dtype: JaxDTypeLike = jnp.float32,
        shd: ShardConfig | None = None,
    ) -> None:
        """Initialize SwiGLU MLP layer.

        Args:
            hidden_size: Hidden dimension of the model.
            intermediate_size: Intermediate projection dimension.
            rngs: Random number generators for weight initialization.
            dtype: Computation data type.
            shd: Optional sharding configuration.

        """
        resolved_shd = shd if shd is not None else ShardConfig.no_sharding()
        fc1_meta = {"out_sharding": resolved_shd.fc1_kernel} if resolved_shd.fc1_kernel is not None else {}
        fc1_bias_meta = {"out_sharding": resolved_shd.fc1_bias} if resolved_shd.fc1_bias is not None else {}
        fc2_meta = {"out_sharding": resolved_shd.fc2_kernel} if resolved_shd.fc2_kernel is not None else {}
        fc2_bias_meta = {"out_sharding": resolved_shd.fc2_bias} if resolved_shd.fc2_bias is not None else {}
        self.gate_proj = make_linear(hidden_size, intermediate_size, use_bias=False, kernel_metadata=fc1_meta, bias_metadata=fc1_bias_meta, rngs=rngs)
        self.up_proj = make_linear(hidden_size, intermediate_size, use_bias=False, kernel_metadata=fc1_meta, bias_metadata=fc1_bias_meta, rngs=rngs)
        self.down_proj = make_linear(intermediate_size, hidden_size, use_bias=False, kernel_metadata=fc2_meta, bias_metadata=fc2_bias_meta, rngs=rngs)
        self.dtype: JaxDTypeLike = dtype

    def inject_lora_projections(
        self,
        target_modules: list[str],
        adapter_factory: Callable[[nnx.Linear], LinearLayerProtocol],
    ) -> int:
        """Wrap matching MLP linear projections (`gate_proj`, `up_proj`, `down_proj`) with LoRA adapters.

        Args:
            target_modules: List of target projection names to wrap.
            adapter_factory: Factory callable converting an `nnx.Linear` into a `LinearLayerProtocol` adapter.

        Returns:
            Number of projections wrapped in this MLP block.

        """
        injected = 0
        if "gate_proj" in target_modules and isinstance(self.gate_proj, nnx.Linear):
            self.gate_proj = adapter_factory(self.gate_proj)
            injected += 1
        if "up_proj" in target_modules and isinstance(self.up_proj, nnx.Linear):
            self.up_proj = adapter_factory(self.up_proj)
            injected += 1
        if "down_proj" in target_modules and isinstance(self.down_proj, nnx.Linear):
            self.down_proj = adapter_factory(self.down_proj)
            injected += 1
        return injected

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
        res: Array = out.astype(self.dtype)
        return res
