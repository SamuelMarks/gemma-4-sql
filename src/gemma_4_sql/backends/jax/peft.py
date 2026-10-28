"""JAX/Flax NNX Parameter-Efficient Fine-Tuning (PEFT / LoRA) implementation."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array

from gemma_4_sql.backends.common_peft import apply_lora_wrapper, compute_lora_scale
from gemma_4_sql.backends.jax.gemma4.config import JaxDTypeLike, ModelConfig, resolve_gemma4_config
from gemma_4_sql.backends.jax.gemma4.decoder_layer import Gemma4DecoderLayer
from gemma_4_sql.backends.jax.gemma4.layers import LinearLayerProtocol
from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM
from gemma_4_sql.backends.jax.inference import MODEL_CACHE as _MODEL_CACHE
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import BackendKwargValue, PEFTResultDict

logger = logging.getLogger(__name__)

Gemma4Config = ModelConfig


@runtime_checkable
class LoRAInjectableProtocol(Protocol):
    """Protocol for modules or layers supporting explicit `inject_lora_projections` traversal."""

    def inject_lora_projections(
        self,
        target_modules: list[str],
        adapter_factory: Callable[[nnx.Linear], LinearLayerProtocol],
    ) -> int:
        """Wrap matching linear projections with `LinearLayerProtocol` adapters."""
        ...


@runtime_checkable
class LayerSequenceContainerProtocol(Protocol):
    """Protocol for model containers exposing a `.layers` sequence of injectable decoder layers."""

    @property
    def layers(self) -> Sequence[LoRAInjectableProtocol]:
        """Provide the sequence of decoder layers."""
        ...


class NNXParam(nnx.Param[Array]):
    """Flax NNX parameter subclass providing uniform `.shape`, `.dtype`, and slice indexing."""


class LoRAParam(NNXParam):
    """Marker parameter class designating low-rank adapter weights in Flax NNX."""


class LoRALinearAdapter(nnx.Module):
    """Low-rank adaptation (LoRA) linear adapter module for Flax NNX implementing `LinearLayerProtocol`.

    Decomposes linear weight updates into low-rank factor matrices:
        y = x W + bias + (alpha / r) * (Dropout(x) @ A) @ B
    where:
        W is the base frozen weight kernel with shape `(in_features, out_features)`,
        A is the down-projection adapter with shape `(in_features, r)`,
        B is the up-projection adapter with shape `(r, out_features)`.
    """

    in_features: int
    out_features: int
    r: int
    lora_alpha: float
    scale: float
    kernel: nnx.Param[Array]
    bias: nnx.Param[Array] | None
    lora_a: LoRAParam
    lora_b: LoRAParam
    dropout: nnx.Dropout | None
    dtype: JaxDTypeLike

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        use_bias: bool = False,
        dtype: JaxDTypeLike | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialize the NNX LoRA linear adapter layer.

        Args:
            in_features: Number of input features (`d_in`).
            out_features: Number of output features (`d_out`).
            r: Rank of the low-rank adapters (`r > 0`).
            lora_alpha: Scaling factor (`alpha`).
            lora_dropout: Dropout probability applied to inputs prior to adapter projection.
            use_bias: Whether to maintain a bias parameter array.
            dtype: Computation and parameter data type (defaults to `jnp.float32`).
            rngs: Flax NNX random number generators.

        Raises:
            DependencyMissingError: If JAX or Flax NNX dependencies are missing.
            ValueError: If rank `r` is less than or equal to 0.

        """
        if nnx is None or jax is None or jnp is None:
            raise DependencyMissingError("JAX PEFT dependencies are missing.")
        self.scale = compute_lora_scale(lora_alpha, r)
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = float(lora_alpha)
        self.dtype = dtype if dtype is not None else jnp.float32

        resolved_rngs = rngs if rngs is not None else nnx.Rngs(0)

        self.kernel = NNXParam(jnp.zeros((in_features, out_features), dtype=self.dtype))
        if use_bias:
            self.bias = NNXParam(jnp.zeros((out_features,), dtype=self.dtype))
        else:
            self.bias = None

        scale_init = 1.0 / math.sqrt(r)
        lora_a_arr = jax.random.uniform(
            resolved_rngs.params(),
            (in_features, r),
            dtype=self.dtype,
            minval=-scale_init,
            maxval=scale_init,
        )
        self.lora_a = LoRAParam(lora_a_arr)
        self.lora_b = LoRAParam(jnp.zeros((r, out_features), dtype=self.dtype))

        if lora_dropout > 0.0:
            self.dropout = nnx.Dropout(rate=lora_dropout, rngs=resolved_rngs)
        else:
            self.dropout = None

    @property
    def W(self) -> Array:
        """Provide the base weight array `W` with shape `(in_features, out_features)`."""
        return self.kernel[...]

    @property
    def A(self) -> Array:
        """Provide the down-projection adapter `A` with shape `(in_features, r)`."""
        return self.lora_a[...]

    @property
    def B(self) -> Array:
        """Provide the up-projection adapter `B` with shape `(r, out_features)`."""
        return self.lora_b[...]

    def train(self, mode: bool = True, **attributes: bool) -> None:
        """Set module training mode to toggle LoRA dropout active or inactive.

        Args:
            mode: `True` to enable training dropout, `False` for inference mode.
            **attributes: Optional Flax NNX module state attributes.

        """
        _ = attributes
        if self.dropout is not None:
            if mode:
                self.dropout.train()
            else:
                self.dropout.eval()

    def eval(self, **attributes: bool) -> None:
        """Set module to evaluation/inference mode disabling LoRA dropout.

        Args:
            **attributes: Optional Flax NNX module state attributes.

        """
        self.train(False, **attributes)

    @classmethod
    def from_linear(
        cls,
        linear: nnx.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> LoRALinearAdapter:
        """Construct a `LoRALinearAdapter` wrapping an existing `nnx.Linear`.

        Args:
            linear: Source `flax.nnx.Linear` module whose weights will be wrapped.
            r: Rank of the low-rank adapters.
            lora_alpha: Scaling parameter.
            lora_dropout: Dropout rate applied prior to adapter projection.
            rngs: Flax NNX random number generators.

        Returns:
            A `LoRALinearAdapter` instance sharing base kernel and bias with `linear`.

        """
        kernel_val = linear.kernel[...]
        in_features, out_features = kernel_val.shape
        bias_param = linear.bias
        use_bias = bias_param is not None
        dtype: JaxDTypeLike = kernel_val.dtype

        lora_layer = cls(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_bias=use_bias,
            dtype=dtype,
            rngs=rngs,
        )
        lora_layer.kernel = linear.kernel
        if use_bias:
            lora_layer.bias = bias_param
        return lora_layer

    def __call__(self, x: Array, *, deterministic: bool | None = None) -> Array:
        """Execute the forward linear projection with low-rank adaptation.

        Computes:
            `y = x @ W + bias + scale * (dropout(x) @ A) @ B`

        Args:
            x: Input array of shape `(..., in_features)`.
            deterministic: Optional override flag to disable dropout during inference.

        Returns:
            Output array of shape `(..., out_features)`.

        """
        base = jnp.dot(x, self.kernel[...])
        if self.bias is not None:
            base = base + self.bias[...]

        dropped_x = self.dropout(x, deterministic=deterministic) if self.dropout is not None else x
        lora_term = jnp.dot(jnp.dot(dropped_x, self.lora_a[...]), self.lora_b[...])
        out: Array = (base + self.scale * lora_term).astype(self.dtype)
        return out


NNXLoRALinear = LoRALinearAdapter


@runtime_checkable
class _AnyLayerObjProtocol(Protocol):
    """Protocol satisfied by any JAX layer, model, or test double."""

    def __repr__(self) -> str:
        """Return string representation."""
        ...


def inject_lora_to_layer(
    layer: LoRAInjectableProtocol | Gemma4DecoderLayer | nnx.Module | _AnyLayerObjProtocol | str,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    *,
    rngs: nnx.Rngs | None = None,
) -> int:
    """Inject LoRA adapters into a single module implementing `LoRAInjectableProtocol`.

    Args:
        layer: Target NNX layer or block.
        target_modules: List of target projection names (`'q_proj'`, `'v_proj'`, etc.).
        lora_r: Rank of the low-rank update matrices.
        lora_alpha: Scaling factor for LoRA updates.
        lora_dropout: Dropout probability before low-rank projection.
        rngs: Optional Flax NNX random number generators.

    Returns:
        The number of adapted projection modules injected into the layer.

    """
    resolved_rngs = rngs if rngs is not None else nnx.Rngs(0)
    if not isinstance(layer, LoRAInjectableProtocol):
        return 0

    def _make_adapter(linear: nnx.Linear) -> LinearLayerProtocol:
        """Wrap an `nnx.Linear` projection with `LoRALinearAdapter`."""
        return LoRALinearAdapter.from_linear(
            linear,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            rngs=resolved_rngs,
        )

    return layer.inject_lora_projections(target_modules, _make_adapter)


def inject_lora[ModelT](
    model: ModelT,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    *,
    rngs: nnx.Rngs | None = None,
) -> tuple[ModelT, int]:
    """Inject LoRA adapters into designated projection modules of a JAX/Flax NNX model.

    Args:
        model: JAX model (such as `Gemma4ForCausalLM`, `LoRAInjectableProtocol`, or `LayerSequenceContainerProtocol`).
        target_modules: List of target submodule names (e.g. `['q_proj', 'v_proj']`).
        lora_r: Rank dimension for LoRA adapters.
        lora_alpha: Scaling coefficient for LoRA.
        lora_dropout: Dropout probability applied before adapter projection.
        rngs: Optional Flax NNX random number generators.

    Returns:
        A tuple of `(adapted_model, total_injected_count)`.

    Raises:
        DependencyMissingError: If JAX or Flax NNX dependencies are missing.

    """
    if nnx is None:
        raise DependencyMissingError("JAX PEFT dependencies are missing.")

    resolved_rngs = rngs if rngs is not None else nnx.Rngs(0)
    if not target_modules:
        return model, 0

    injected = inject_lora_to_layer(
        model,
        target_modules=target_modules,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        rngs=resolved_rngs,
    )
    return model, injected


def create_lora_optimizer[ModelT: nnx.Module](
    model: ModelT,
    tx: optax.GradientTransformation | None = None,
) -> nnx.Optimizer[ModelT]:
    """Create a Flax NNX Optimizer configured to update only LoRA adapter parameters.

    Args:
        model: Flax NNX model with injected LoRA adapters.
        tx: Optional Optax gradient transformation (defaults to `optax.adam(1e-4)`).

    Returns:
        An `nnx.Optimizer` bound to update only `LoRAParam` parameters.

    Raises:
        DependencyMissingError: If Flax NNX or Optax dependencies are missing.

    """
    if nnx is None or optax is None:
        raise DependencyMissingError("JAX PEFT dependencies are missing.")

    resolved_tx = tx if tx is not None else optax.adam(1e-4)
    return nnx.Optimizer(model, resolved_tx, wrt=LoRAParam)


def count_parameters(model: nnx.Module) -> tuple[int, int]:
    """Count total and trainable LoRA parameters in a Flax NNX model.

    Args:
        model: Flax NNX model with LoRA adapters.

    Returns:
        A tuple of `(total_parameters_count, trainable_lora_parameters_count)`.

    Raises:
        DependencyMissingError: If JAX or Flax NNX dependencies are missing.

    """
    if nnx is None or jax is None:
        raise DependencyMissingError("JAX PEFT dependencies are missing.")

    _, lora_state, base_state = nnx.split(model, LoRAParam, ...)
    lora_count = sum(int(x.size) for x in jax.tree.leaves(lora_state))
    base_count = sum(int(x.size) for x in jax.tree.leaves(base_state))
    return lora_count + base_count, lora_count


def apply_lora(
    model_name: str,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    **kwargs: BackendKwargValue | ModelConfig | nnx.Rngs | Gemma4ForCausalLM,
) -> PEFTResultDict:
    """Apply LoRA to a model using the JAX backend.

    Args:
        model_name: The name of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.
        **kwargs: Optional keyword arguments, including `'model'`, `'config'`, `'rngs'`, and `'raise_on_error'`.

    Returns:
        A dictionary containing the PEFT operation results.

    Raises:
        DependencyMissingError: If JAX PEFT dependencies are missing.

    """
    if optax is None or jax is None or nnx is None or Gemma4ForCausalLM is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise DependencyMissingError("JAX PEFT dependencies are missing.")

    raw_rngs = kwargs.get("rngs")
    resolved_rngs = raw_rngs if isinstance(raw_rngs, nnx.Rngs) else nnx.Rngs(0)
    raw_model = kwargs.get("model")
    raw_config = kwargs.get("config")
    raise_on_error = bool(kwargs.get("raise_on_error"))

    def _inject() -> int:
        """Instantiate or reuse JAX model and inject LoRA layers."""
        if isinstance(raw_model, Gemma4ForCausalLM):
            model_obj = raw_model
        else:
            cfg = raw_config if isinstance(raw_config, ModelConfig) else resolve_gemma4_config(Gemma4Config, model_name)
            model_obj = Gemma4ForCausalLM(cfg, rngs=resolved_rngs)

        model, injected_count = inject_lora(
            model=model_obj,
            target_modules=target_modules,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            rngs=resolved_rngs,
        )
        _ = nnx.split(model, LoRAParam, ...)
        _MODEL_CACHE.put_model(model_name, model)
        logger.info("Injected LoRA into %d targets", injected_count)
        return injected_count

    return apply_lora_wrapper(
        "jax",
        model_name,
        target_modules,
        lora_r,
        lora_alpha,
        lora_dropout,
        _inject,
        raise_on_error=raise_on_error,
    )


__all__ = [
    "Gemma4Config",
    "Gemma4DecoderLayer",
    "Gemma4ForCausalLM",
    "LayerSequenceContainerProtocol",
    "LinearLayerProtocol",
    "LoRAInjectableProtocol",
    "LoRALinearAdapter",
    "LoRAParam",
    "NNXLoRALinear",
    "NNXParam",
    "apply_lora",
    "count_parameters",
    "create_lora_optimizer",
    "inject_lora",
    "inject_lora_to_layer",
]
