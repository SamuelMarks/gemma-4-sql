"""Keras-specific PEFT / LoRA implementation."""

from __future__ import annotations

import contextlib
import logging
import math
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, SupportsFloat, runtime_checkable

import keras
from keras import ops

from gemma_4_sql.backends.common_peft import apply_lora_wrapper, compute_lora_scale, matches_target_module
from gemma_4_sql.backends.lazy_loader import optional_attr
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    KerasBackboneProtocol,
    KerasCausalLMProtocol,
    KerasLayerProtocol,
    KerasVariableProtocol,
    PEFTResultDict,
    TensorType,
    TfTensorProtocol,
    register_quantized_artifact,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class KerasDenseLayerProtocol(KerasLayerProtocol, Protocol):
    """Protocol for a Keras `Dense` layer supporting `.kernel`, `.bias`, `.built`, and `.build(...)`."""

    built: bool

    @property
    def kernel(self) -> KerasVariableProtocol:
        """Provide the Dense kernel weight variable."""
        ...

    @property
    def bias(self) -> KerasVariableProtocol | None:
        """Provide the optional Dense bias variable."""
        ...

    def build(self, input_shape: tuple[int | None, ...] | None = None) -> None:
        """Build the Dense layer variables."""
        ...

    def __call__(self, inputs: TensorType) -> TfTensorProtocol:
        """Apply the Dense projection to `inputs`."""
        ...


@runtime_checkable
class _BackboneModelProtocol(Protocol):
    """Protocol for a KerasNLP model exposing `.backbone`."""

    @property
    def backbone(self) -> KerasBackboneProtocol:
        """Provide the underlying `KerasBackboneProtocol`."""
        ...


@runtime_checkable
class _GemmaCausalLMLoaderProtocol(Protocol):
    """Protocol for `keras_nlp.models.GemmaCausalLM` exposing `.from_preset(model_name)`."""

    def from_preset(self, model_name: str, /) -> KerasCausalLMProtocol | _BackboneModelProtocol:
        """Load a `GemmaCausalLM` model from a preset identifier."""
        ...


class KerasLoRADense(keras.layers.Layer):
    """Low-rank adaptation (LoRA) layer wrapper for Keras Dense layers.

    Decomposes the weight update of a Dense layer into low-rank factor matrices:
        y = W x + bias + (alpha / r) * (Dropout(x) @ A) @ B
    where:
        W is the base frozen weight kernel with shape `(in_features, units)`,
        A is the down-projection adapter with shape `(in_features, r)`,
        B is the up-projection adapter with shape `(r, units)`.
    """

    dense: KerasDenseLayerProtocol
    r: int
    lora_alpha: float
    scale: float
    lora_dropout: float
    dropout: keras.layers.Dropout | None
    lora_a: KerasVariableProtocol
    lora_b: KerasVariableProtocol
    _lora_built: bool

    def __init__(
        self,
        dense: KerasDenseLayerProtocol,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        *,
        name: str | None = None,
        trainable: bool = True,
    ) -> None:
        """Initialize the Keras LoRA Dense layer wrapper.

        Args:
            dense: The base Keras Dense layer to adapt.
            r: Rank of the low-rank adaptation matrices (`r > 0`).
            lora_alpha: Scaling parameter (`alpha`).
            lora_dropout: Dropout probability applied to inputs prior to LoRA projection.
            name: Optional layer name.
            trainable: Whether the layer is trainable.

        Raises:
            DependencyMissingError: If Keras is missing.
            ValueError: If rank `r` is less than or equal to 0.

        """
        if keras is None or ops is None:
            raise DependencyMissingError("Keras dependencies are missing.")
        self.scale = compute_lora_scale(lora_alpha, r)

        super().__init__(name=name, trainable=trainable)
        self.dense = dense
        self.dense.trainable = False
        self.r = r
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = lora_dropout
        self._lora_built = False
        if lora_dropout > 0.0:
            self.dropout = keras.layers.Dropout(rate=lora_dropout)
        else:
            self.dropout = None

        with contextlib.suppress(AttributeError):
            if self.dense.built:
                self.build()

    def build(self, input_shape: tuple[int | None, ...] | None = None) -> None:
        """Build the LoRA layer parameters.

        Args:
            input_shape: Shape tuple of the incoming tensor.

        """
        if self._lora_built:
            return

        if not self.dense.built and input_shape is not None:
            self.dense.build(input_shape)

        in_features = int(self.dense.kernel.shape[0])
        units = int(self.dense.kernel.shape[1])
        scale_init = 1.0 / math.sqrt(self.r)

        self.lora_a = self.add_weight(
            shape=(in_features, self.r),
            initializer=keras.initializers.RandomUniform(minval=-scale_init, maxval=scale_init),
            trainable=True,
            name="lora_a",
        )
        self.lora_b = self.add_weight(
            shape=(self.r, units),
            initializer="zeros",
            trainable=True,
            name="lora_b",
        )
        self._lora_built = True
        super().build(input_shape)

    def __call__[InputT](self, inputs: InputT, training: bool | None = None) -> InputT:
        """Invoke the layer forward pass, building LoRA weights lazily if needed."""
        if not self._lora_built:
            self.build(tuple(inputs.shape))  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]  # ty: ignore[unresolved-attribute]
        return self.call(inputs, training=training)  # pyright: ignore[reportArgumentType,reportReturnType]  # ty: ignore[invalid-return-type, invalid-argument-type]

    def call(self, inputs: TensorType | SupportsFloat, training: bool | None = None) -> TfTensorProtocol:
        """Execute the forward pass combining frozen base projection and low-rank adapter.

        Args:
            inputs: Input tensor.
            training: Boolean indicating training or inference mode for dropout.

        Returns:
            Output tensor of shape `(..., units)`.

        """
        base_out = self.dense(inputs)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
        dropped = self.dropout(inputs, training=training) if self.dropout is not None else inputs
        lora_out: TfTensorProtocol = ops.matmul(ops.matmul(dropped, self.lora_a), self.lora_b)  # pyright: ignore[reportAssignmentType]
        res: TfTensorProtocol = base_out + self.scale * lora_out  # pyright: ignore[reportAssignmentType]
        return res

    @property
    def kernel(self) -> KerasVariableProtocol:
        """Provide the base layer kernel weight."""
        return self.dense.kernel

    @property
    def bias(self) -> KerasVariableProtocol | None:
        """Provide the base layer bias weight if present."""
        return self.dense.bias

    @property
    def W(self) -> KerasVariableProtocol:
        """Provide alias for base weight kernel."""
        return self.dense.kernel

    @property
    def A(self) -> KerasVariableProtocol:
        """Provide alias for down-projection adapter matrix."""
        return self.lora_a

    @property
    def B(self) -> KerasVariableProtocol:
        """Provide alias for up-projection adapter matrix."""
        return self.lora_b

    def merge_weights(self) -> KerasDenseLayerProtocol:
        """Fold low-rank adapter weights into the base Dense kernel and return the base layer.

        Computes:
            `W_merged = W + scale * (lora_a @ lora_b)`
        and assigns it to the underlying Dense layer kernel.

        Returns:
            The original Keras Dense layer with updated weights.

        """
        if not self._lora_built:
            return self.dense

        raw_matmul: TfTensorProtocol = ops.matmul(self.lora_a, self.lora_b)  # pyright: ignore[reportAssignmentType]
        delta = self.scale * raw_matmul
        self.dense.kernel.assign(self.dense.kernel + delta)  # pyright: ignore[reportOperatorIssue]  # ty: ignore[unsupported-operator]
        self.dense.trainable = True
        return self.dense


@dataclass(frozen=True)
class KerasLoRALayerState:
    """State record for a single Keras `Dense` layer adapted by `KerasLoRAAdapter`."""

    layer_name: str
    base_layer: KerasDenseLayerProtocol
    lora_layer: KerasLoRADense


@dataclass
class KerasLoRAAdapter:
    """Typed container tracking all `KerasLoRALayerState` entries injected into a Keras model."""

    layers: dict[str, KerasLoRALayerState]
    rank: int
    alpha: float
    dropout: float

    def merge_all(self) -> int:
        """Merge all tracked `KerasLoRADense` layers back into their underlying `Dense` kernels.

        Returns:
            Count of merged adapter layers.

        """
        for state in self.layers.values():
            state.lora_layer.merge_weights()
        return len(self.layers)


_KERAS_LORA_REGISTRY: weakref.WeakKeyDictionary[KerasCausalLMProtocol | KerasLayerProtocol, KerasLoRAAdapter] = weakref.WeakKeyDictionary()
_KERAS_LORA_ID_FALLBACK: dict[int, KerasLoRAAdapter] = {}


@runtime_checkable
class _LayersContainerProtocol(Protocol):
    """Protocol for any Keras model or container exposing `.layers`."""

    @property
    def layers(self) -> Sequence[KerasLayerProtocol]:
        """Provide the contained layers."""
        ...


def get_keras_lora_adapter(
    model: KerasCausalLMProtocol | KerasLayerProtocol | _LayersContainerProtocol | _BackboneModelProtocol | Sequence[KerasLayerProtocol] | str | None,
) -> KerasLoRAAdapter | None:
    """Look up the registered `KerasLoRAAdapter` for a model instance.

    Args:
        model: Target Keras model or layer container.

    Returns:
        Registered `KerasLoRAAdapter` or `None` if no LoRA adapters were injected.

    """
    with contextlib.suppress(TypeError):
        found = _KERAS_LORA_REGISTRY.get(model)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
        if found is not None:
            return found
    return _KERAS_LORA_ID_FALLBACK.get(id(model))


def _collect_dense_layers(
    model: KerasCausalLMProtocol | KerasLayerProtocol | Sequence[KerasLayerProtocol],
) -> list[KerasDenseLayerProtocol]:
    """Collect all `keras.layers.Dense` layers from `model.layers` or a sequence of layers.

    Args:
        model: Keras model or layer container exposing `.layers`.

    Returns:
        Deduplicated list of `keras.layers.Dense` instances.

    """
    seen: set[int] = set()
    collected: list[KerasDenseLayerProtocol] = []

    def _visit_seq(seq: Sequence[KerasLayerProtocol]) -> None:
        """Visit a sequence of Keras layers and collect `Dense` instances."""
        for item in seq:
            item_id = id(item)
            if item_id in seen:
                continue
            seen.add(item_id)
            if isinstance(item, keras.layers.Dense):
                collected.append(item)
            with contextlib.suppress(AttributeError, TypeError):
                sub_layers = item.layers  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
                _visit_seq(sub_layers)

    if isinstance(model, (list, tuple)):
        _visit_seq(model)
    else:
        with contextlib.suppress(AttributeError, TypeError):
            top_layers = model.layers  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
            _visit_seq(top_layers)
    return collected


def inject_lora[ModelT](
    model: ModelT,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> tuple[ModelT, int]:
    """Inject `KerasLoRADense` adapter wrappers into target `Dense` layers of a Keras model.

    Tracks adapted layers in a typed `KerasLoRAAdapter` (`dict[str, KerasLoRALayerState]`)
    and registers metadata via `register_quantized_artifact` and `get_keras_lora_adapter`.

    Args:
        model: Target Keras model or layer container exposing `.layers`.
        target_modules: List of target layer names to wrap with LoRA.
        lora_r: Rank dimension of LoRA adapters.
        lora_alpha: Scaling factor for LoRA.
        lora_dropout: Dropout probability applied before adapter projection.

    Returns:
        A tuple of `(adapted_model, count_of_injected_adapters)`.

    Raises:
        DependencyMissingError: If Keras is not available.

    """
    if keras is None:
        raise DependencyMissingError("Keras dependencies are missing.")

    if not target_modules:
        return model, 0

    dense_layers = _collect_dense_layers(model)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
    adapted_states: dict[str, KerasLoRALayerState] = {}

    for dense in dense_layers:
        layer_name = str(dense.name)
        if matches_target_module(layer_name, target_modules):
            lora_layer = KerasLoRADense(
                dense=dense,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            adapted_states[layer_name] = KerasLoRALayerState(
                layer_name=layer_name,
                base_layer=dense,
                lora_layer=lora_layer,
            )
        else:
            dense.trainable = False

    adapter = KerasLoRAAdapter(
        layers=adapted_states,
        rank=lora_r,
        alpha=float(lora_alpha),
        dropout=lora_dropout,
    )
    with contextlib.suppress(TypeError):
        _KERAS_LORA_REGISTRY[model] = adapter  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-assignment]
    _KERAS_LORA_ID_FALLBACK[id(model)] = adapter
    register_quantized_artifact(
        model,  # pyright: ignore[reportArgumentType]
        method="lora",
        scales={name: state.lora_layer.scale for name, state in adapted_states.items()},
    )

    return model, len(adapted_states)


def merge_lora_weights[ModelT](model: ModelT) -> ModelT:
    """Fold all tracked `KerasLoRADense` adapter weights back into base `Dense` kernels on `model`.

    Args:
        model: Keras model registered with `KerasLoRAAdapter` or containing `KerasLoRADense` layers.

    Returns:
        The model with all LoRA layers folded back into native `Dense` kernels.

    """
    adapter = get_keras_lora_adapter(model)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
    if adapter is not None:
        adapter.merge_all()
        return model

    with contextlib.suppress(AttributeError, TypeError):
        for layer in (l for l in model.layers if isinstance(l, KerasLoRADense)):  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
            layer.merge_weights()
    return model


def count_parameters(model: KerasLayerProtocol | KerasCausalLMProtocol) -> tuple[int, int]:
    """Count total and trainable weights in a Keras model.

    Args:
        model: Keras model or layer exposing `.weights` and `.trainable_weights`.

    Returns:
        A tuple of `(total_variable_weights_count, trainable_weights_count)`.

    """
    total = 0
    trainable = 0
    with contextlib.suppress(AttributeError, TypeError):
        total = sum(int(ops.size(w)) for w in model.weights)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]  # ty: ignore[unresolved-attribute]
    with contextlib.suppress(AttributeError, TypeError):
        trainable = sum(int(ops.size(w)) for w in model.trainable_weights)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]  # ty: ignore[unresolved-attribute]
    return total, trainable


def apply_lora(
    model_name: str,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    **kwargs: BackendKwargValue | KerasCausalLMProtocol | KerasLayerProtocol | _BackboneModelProtocol | _LayersContainerProtocol,
) -> PEFTResultDict:
    """Apply LoRA to a model using the Keras backend.

    Args:
        model_name: The name of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.
        **kwargs: Optional keyword arguments, including `'model'`, `'merge'`, and `'raise_on_error'`.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If Keras dependencies are missing.
        TrainingError: If LoRA injection fails and `raise_on_error` is enabled.

    """
    if keras is None:
        raise DependencyMissingError("Keras dependencies are missing.")

    def _inject() -> int:
        """Load or reuse Keras model and inject or merge LoRA weights."""
        provided_model = kwargs.get("model")
        if provided_model is not None:
            model = provided_model
        else:
            gemma_causal_lm_cls = optional_attr(
                "keras_nlp.models",
                "GemmaCausalLM",
                _GemmaCausalLMLoaderProtocol,
            )
            if gemma_causal_lm_cls is None:
                raise ImportError("keras_nlp.models.GemmaCausalLM is not available.")
            model = gemma_causal_lm_cls.from_preset(model_name)

        backbone: KerasBackboneProtocol | None = None
        with contextlib.suppress(AttributeError):
            backbone = model.backbone  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]

        if backbone is not None:
            backbone.enable_lora(rank=lora_r)
            with contextlib.suppress(AttributeError, TypeError):
                for layer in backbone.layers:
                    with contextlib.suppress(AttributeError):
                        layer.trainable = bool(layer.trainable_variables)  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
            logger.info("Enabled Keras native LoRA with rank %d", lora_r)
            injected_count = len(target_modules)
        else:
            model, injected_count = inject_lora(
                model=model,
                target_modules=target_modules,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )

        if kwargs.get("merge"):
            _ = merge_lora_weights(model)
        return injected_count

    wrapper_kwargs: dict[str, StatusValue] = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None}
    return apply_lora_wrapper(
        "keras",
        model_name,
        target_modules,
        lora_r,
        lora_alpha,
        lora_dropout,
        _inject,
        **wrapper_kwargs,
    )


__all__ = [
    "KerasDenseLayerProtocol",
    "KerasLoRAAdapter",
    "KerasLoRADense",
    "KerasLoRALayerState",
    "apply_lora",
    "count_parameters",
    "get_keras_lora_adapter",
    "inject_lora",
    "merge_lora_weights",
]
