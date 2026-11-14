"""Keras-specific PEFT / LoRA implementation using WeakKeyDictionary without memory leaks."""

from __future__ import annotations

import logging
import math
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, overload, runtime_checkable

from gemma_4_sql.backends.common_peft import apply_lora_wrapper, compute_lora_scale, matches_target_module
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
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
class _KerasDropoutLayerProtocol(Protocol):
    """Protocol for an instantiated `keras.layers.Dropout` layer."""

    def __call__(
        self,
        inputs: TensorType | _KerasTensorInputProtocol,
        *,
        training: bool | None = None,
    ) -> TensorType | _KerasTensorInputProtocol:
        """Apply dropout to `inputs`.

        Args:
            inputs: Input activation tensor.
            training: Training mode flag.

        Returns:
            Dropped-out activation tensor.

        """
        ...


@runtime_checkable
class _KerasLayersNamespaceProtocol(Protocol):
    """Protocol for `keras.layers` exposing `Layer`, `Dense`, and `Dropout`."""

    @property
    def Layer(self) -> type:
        """Provide `keras.layers.Layer` base class."""
        ...

    @property
    def Dense(self) -> type:
        """Provide `keras.layers.Dense` class."""
        ...

    def Dropout(self, *, rate: float) -> _KerasDropoutLayerProtocol:
        """Construct a Keras `Dropout` layer.

        Args:
            rate: Dropout probability.

        Returns:
            Initialized `_KerasDropoutLayerProtocol`.

        """
        ...


@runtime_checkable
class _KerasInitializerProtocol(Protocol):
    """Structural protocol satisfied by Keras weight initializers."""


@runtime_checkable
class _KerasInitializersNamespaceProtocol(Protocol):
    """Protocol for `keras.initializers` exposing `RandomUniform`."""

    def RandomUniform(self, *, minval: float, maxval: float) -> _KerasInitializerProtocol:
        """Construct a `RandomUniform` weight initializer.

        Args:
            minval: Lower uniform bound.
            maxval: Upper uniform bound.

        Returns:
            Configured `_KerasInitializerProtocol`.

        """
        ...


@runtime_checkable
class _KerasPEFTModuleProtocol(Protocol):
    """Protocol for `keras` module exposing `.layers` and `.initializers`."""

    @property
    def layers(self) -> _KerasLayersNamespaceProtocol:
        """Provide `keras.layers` namespace."""
        ...

    @property
    def initializers(self) -> _KerasInitializersNamespaceProtocol:
        """Provide `keras.initializers` namespace."""
        ...


keras: _KerasPEFTModuleProtocol | None = optional_import("keras", _KerasPEFTModuleProtocol)
ops = optional_import("keras.ops")


@runtime_checkable
class _KerasOpsMatmulProtocol(Protocol):
    """Protocol for `keras.ops` providing `matmul` returning `TfTensorProtocol`."""

    def matmul(
        self,
        x1: TensorType | KerasVariableProtocol | _KerasTensorInputProtocol,
        x2: TensorType | KerasVariableProtocol,
        /,
    ) -> TfTensorProtocol:
        """Compute matrix multiplication of `x1` and `x2`.

        Args:
            x1: Left-hand matrix or activation tensor.
            x2: Right-hand projection matrix.

        Returns:
            Matrix product tensor.

        """
        ...


def _typed_ops() -> _KerasOpsMatmulProtocol:
    """Return `keras.ops` narrowed to `_KerasOpsMatmulProtocol`.

    Returns:
        Typed `_KerasOpsMatmulProtocol` view of `keras.ops`.

    """
    assert isinstance(ops, _KerasOpsMatmulProtocol)
    return ops


@runtime_checkable
class KerasDenseLayerProtocol(KerasLayerProtocol, Protocol):
    """Protocol for a Keras `Dense` layer supporting `.kernel`, `.bias`, `.built`, and `.build(...)`."""

    built: bool

    @property
    def kernel(self) -> KerasVariableProtocol:
        """Provide the Dense kernel weight variable.

        Returns:
            Base Dense kernel weight variable.

        """
        ...

    @property
    def bias(self) -> KerasVariableProtocol | None:
        """Provide the optional Dense bias variable.

        Returns:
            Base Dense bias variable or `None`.

        """
        ...

    def build(self, input_shape: tuple[int | None, ...] | None = None) -> None:
        """Build the Dense layer variables.

        Args:
            input_shape: Optional input tensor shape tuple.

        """
        ...

    def __call__(self, inputs: TensorType | _KerasTensorInputProtocol) -> TfTensorProtocol:
        """Apply the Dense projection to `inputs`.

        Args:
            inputs: Input activation tensor.

        Returns:
            Projected Dense output tensor.

        """
        ...


@runtime_checkable
class _BackboneModelProtocol(Protocol):
    """Protocol for a KerasNLP model exposing `.backbone`."""

    @property
    def backbone(self) -> KerasBackboneProtocol:
        """Provide the underlying `KerasBackboneProtocol`.

        Returns:
            Underlying model backbone.

        """
        ...


@runtime_checkable
class _KerasTensorInputProtocol(Protocol):
    """Protocol for any Keras tensor or `OpenVINOKerasTensor` exposing `.shape`."""

    @property
    def shape(self) -> Sequence[int | None] | None:
        """Provide the tensor shape dimensions.

        Returns:
            Sequence of integer or `None` dimension sizes, or `None`.

        """
        ...


@runtime_checkable
class _LayersContainerProtocol(Protocol):
    """Runtime-checkable protocol for any Keras model or container exposing `.layers`."""

    @property
    def layers(self) -> Sequence[KerasLayerProtocol | _LayersContainerProtocol]:
        """Provide the contained layers.

        Returns:
            Sequence of child Keras layers or nested layer containers.

        """
        ...


@runtime_checkable
class _LayerSlotSequence(Protocol):
    """Protocol for mutable layer sequences supporting index assignment `seq[idx] = layer`."""

    def __setitem__(self, index: int, value: KerasLayerProtocol, /) -> None:
        """Assign `value` into container slot `index`.

        Args:
            index: Zero-based slot index.
            value: Replacement `KerasLayerProtocol` layer.

        """
        ...


type KerasPEFTTarget = (
    KerasCausalLMProtocol
    | KerasLayerProtocol
    | KerasDenseLayerProtocol
    | _BackboneModelProtocol
    | _LayersContainerProtocol
    | Sequence[KerasLayerProtocol | _LayersContainerProtocol]
)


@runtime_checkable
class _GemmaCausalLMLoaderProtocol(Protocol):
    """Protocol for `keras_nlp.models.GemmaCausalLM` exposing `.from_preset(model_name)`."""

    def from_preset(self, model_name: str, /) -> KerasCausalLMProtocol | _BackboneModelProtocol:
        """Load a `GemmaCausalLM` model from a preset identifier.

        Args:
            model_name: Preset identifier string.

        Returns:
            Loaded `KerasCausalLMProtocol` or `_BackboneModelProtocol` instance.

        """
        ...


_NATIVE_KERAS_LORA_CLS_CACHE: list[type[KerasLoRADense]] = []


def _get_native_keras_lora_cls() -> type[KerasLoRADense]:
    """Return the cached `KerasLoRADense` subclass inheriting `keras.layers.Layer`.

    Returns:
        Subclass of `(KerasLoRADense, keras.layers.Layer)`.

    """
    if _NATIVE_KERAS_LORA_CLS_CACHE:
        return _NATIVE_KERAS_LORA_CLS_CACHE[0]
    from keras.layers import Layer as _RealKerasLayer

    class _NativeKerasLoRADense(KerasLoRADense, _RealKerasLayer):
        """Concrete `KerasLoRADense` bound to `keras.layers.Layer` for Keras variable tracking."""

    _NATIVE_KERAS_LORA_CLS_CACHE.append(_NativeKerasLoRADense)
    return _NativeKerasLoRADense


class KerasLoRADense:
    """Low-rank adaptation (LoRA) layer wrapper for Keras Dense layers.

    Decomposes the weight update of a Dense layer into low-rank factor matrices:
        `y = W x + bias + (alpha / r) * (Dropout(x) @ A) @ B`

    Attributes:
        name: Layer identifier string.
        trainable: Whether adapter weights are trainable.
        built: Whether base and adapter weights are built.
        weights: Tracked Keras layer variables.
        trainable_weights: Tracked trainable Keras variables.
        trainable_variables: Tracked trainable Keras variables.
        dense: Wrapped frozen base `Dense` layer.
        r: Rank of the LoRA factor matrices.
        lora_alpha: Scaling factor numerator.
        scale: Computed `lora_alpha / r` multiplier.
        lora_dropout: Dropout rate applied on the LoRA branch.
        dropout: Optional `keras.layers.Dropout` layer when `lora_dropout > 0`.
        lora_a: Down-projection weight variable `(in_features, r)`.
        lora_b: Up-projection weight variable `(r, units)`.

    """

    name: str
    built: bool
    dense: KerasDenseLayerProtocol
    r: int
    lora_alpha: float
    scale: float
    lora_dropout: float
    dropout: _KerasDropoutLayerProtocol | None
    lora_a: KerasVariableProtocol
    lora_b: KerasVariableProtocol
    _trainable: bool
    _lora_built: bool

    def __new__(
        cls,
        dense: KerasDenseLayerProtocol,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        *,
        name: str | None = None,
        trainable: bool = True,
    ) -> KerasLoRADense:
        """Allocate a `keras.layers.Layer`-backed `KerasLoRADense` instance.

        Args:
            dense: The base Keras Dense layer to adapt.
            r: Rank of the low-rank adaptation matrices (`r > 0`).
            lora_alpha: Scaling parameter (`alpha`).
            lora_dropout: Dropout probability applied to inputs prior to LoRA projection.
            name: Optional layer name.
            trainable: Whether the layer is trainable.

        Returns:
            Allocated `KerasLoRADense` instance.

        Raises:
            DependencyMissingError: If Keras is missing.
            ValueError: If rank `r` is less than or equal to 0.

        """
        _ = (dense, lora_dropout, name, trainable)
        if keras is None or ops is None:
            raise DependencyMissingError("Keras dependencies are missing.")
        _ = compute_lora_scale(lora_alpha, r)
        native_cls = _get_native_keras_lora_cls()
        return super(KerasLoRADense, native_cls).__new__(native_cls)

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

        """
        assert keras is not None
        self.scale = compute_lora_scale(lora_alpha, r)
        self._trainable = bool(trainable)
        self._lora_built = False

        layer_init: Callable[..., None] = vars(keras.layers.Layer)["__init__"]
        layer_init(self, name=name, trainable=trainable)
        self.dense = dense
        self.dense.trainable = False
        self.r = r
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = lora_dropout
        if lora_dropout > 0.0:
            self.dropout = keras.layers.Dropout(rate=lora_dropout)
        else:
            self.dropout = None

        if bool(self.dense.built):
            self.build()

    @property
    def trainable(self) -> bool:
        """Indicate whether adapter weights receive gradient updates.

        Returns:
            Boolean trainability flag.

        """
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool) -> None:
        """Update adapter trainability while keeping the wrapped base `Dense` frozen.

        Args:
            value: Desired trainability flag for LoRA adapter weights.

        """
        self._trainable = bool(value)

    @property
    def weights(self) -> list[KerasVariableProtocol]:
        """Provide all base and LoRA variables belonging to the layer.

        Returns:
            List of layer weight variables.

        """
        base_weights = list(self.dense.weights)
        if self._lora_built:
            return [*base_weights, self.lora_a, self.lora_b]
        return base_weights

    @property
    def trainable_weights(self) -> list[KerasVariableProtocol]:
        """Provide trainable LoRA adapter variables when `trainable` is `True`.

        Returns:
            List of trainable LoRA adapter variables.

        """
        if self._lora_built and self._trainable:
            return [self.lora_a, self.lora_b]
        return []

    @property
    def trainable_variables(self) -> list[KerasVariableProtocol]:
        """Provide trainable LoRA adapter variables when `trainable` is `True`.

        Returns:
            List of trainable LoRA adapter variables.

        """
        return self.trainable_weights

    def build(self, input_shape: tuple[int | None, ...] | None = None) -> None:
        """Build the LoRA layer parameters.

        Args:
            input_shape: Shape tuple of the incoming tensor.

        """
        if self._lora_built:
            return
        assert keras is not None

        if not self.dense.built and input_shape is not None:
            self.dense.build(input_shape)

        in_features = int(self.dense.kernel.shape[0])
        units = int(self.dense.kernel.shape[1])
        scale_init = 1.0 / math.sqrt(self.r)

        add_weight_fn: Callable[..., KerasVariableProtocol] = vars(keras.layers.Layer)["add_weight"]
        self.lora_a = add_weight_fn(
            self,
            shape=(in_features, self.r),
            initializer=keras.initializers.RandomUniform(minval=-scale_init, maxval=scale_init),
            trainable=True,
            name="lora_a",
        )
        self.lora_b = add_weight_fn(
            self,
            shape=(self.r, units),
            initializer="zeros",
            trainable=True,
            name="lora_b",
        )
        self._lora_built = True
        layer_build: Callable[[KerasLoRADense, tuple[int | None, ...] | None], None] = vars(keras.layers.Layer)["build"]
        layer_build(self, input_shape)

    @overload
    def __call__(self, inputs: TfTensorProtocol, training: bool | None = None) -> TfTensorProtocol:
        """Invoke the layer forward pass on a `TfTensorProtocol` tensor."""

    @overload
    def __call__(
        self, inputs: TensorType | _KerasTensorInputProtocol, training: bool | None = None
    ) -> TfTensorProtocol:
        """Invoke the layer forward pass on a generic Keras tensor."""

    def __call__(
        self,
        inputs: TensorType | _KerasTensorInputProtocol,
        training: bool | None = None,
    ) -> TfTensorProtocol:
        """Invoke the layer forward pass, building LoRA weights lazily if needed.

        Args:
            inputs: Incoming activation tensor.
            training: Optional training flag for dropout behavior.

        Returns:
            Output `TfTensorProtocol` tensor of shape `(..., units)`.

        """
        if not self._lora_built:
            raw_shape = inputs.shape
            self.build(tuple(raw_shape) if raw_shape is not None else None)
        return self.call(inputs, training=training)

    def call(self, inputs: TensorType | _KerasTensorInputProtocol, training: bool | None = None) -> TfTensorProtocol:
        """Execute the forward pass combining frozen base projection and low-rank adapter.

        Args:
            inputs: Input tensor.
            training: Boolean indicating training or inference mode for dropout.

        Returns:
            Output tensor of shape `(..., units)`.

        """
        base_out = self.dense(inputs)
        dropped: TensorType | _KerasTensorInputProtocol = (
            self.dropout(inputs, training=training) if self.dropout is not None else inputs
        )
        k_ops = _typed_ops()
        lora_out: TfTensorProtocol = k_ops.matmul(k_ops.matmul(dropped, self.lora_a), self.lora_b)
        scaled_lora: TfTensorProtocol = lora_out * self.scale
        res: TfTensorProtocol = base_out + scaled_lora
        return res

    @property
    def kernel(self) -> KerasVariableProtocol:
        """Base layer kernel weight."""
        return self.dense.kernel

    @property
    def bias(self) -> KerasVariableProtocol | None:
        """Base layer bias weight if present."""
        return self.dense.bias

    @property
    def W(self) -> KerasVariableProtocol:
        """Alias for base weight kernel."""
        return self.dense.kernel

    @property
    def A(self) -> KerasVariableProtocol:
        """Alias for down-projection adapter matrix."""
        return self.lora_a

    @property
    def B(self) -> KerasVariableProtocol:
        """Alias for up-projection adapter matrix."""
        return self.lora_b

    def merge_weights(self) -> KerasDenseLayerProtocol:
        """Fold low-rank adapter weights into the base Dense kernel and return the base layer.

        Computes `W_merged = W + scale * (lora_a @ lora_b)` and assigns it to the
        underlying Dense layer kernel.

        Returns:
            The original Keras Dense layer with updated weights.

        """
        if not self._lora_built:
            return self.dense

        k_ops = _typed_ops()
        raw_matmul: TfTensorProtocol = k_ops.matmul(self.lora_a, self.lora_b)
        delta: TfTensorProtocol = raw_matmul * self.scale
        self.dense.kernel.assign(self.dense.kernel + delta)
        self.dense.trainable = True
        return self.dense


@dataclass(frozen=True)
class KerasLoRALayerState:
    """State record for a single Keras `Dense` layer adapted by `KerasLoRAAdapter`.

    Attributes:
        layer_name: Name of the wrapped `Dense` layer.
        base_layer: Underlying frozen `Dense` layer.
        lora_layer: Injected `KerasLoRADense` wrapper.
        container_slots: Mutable container references and indices holding `lora_layer`.

    """

    layer_name: str
    base_layer: KerasDenseLayerProtocol
    lora_layer: KerasLoRADense
    container_slots: tuple[tuple[_LayerSlotSequence, int], ...] = ()


@dataclass
class KerasLoRAAdapter:
    """Typed container tracking all `KerasLoRALayerState` entries injected into a Keras model.

    Attributes:
        layers: Mapping from layer name to `KerasLoRALayerState`.
        rank: LoRA rank `r`.
        alpha: LoRA scaling factor `alpha`.
        dropout: LoRA dropout rate.

    """

    layers: dict[str, KerasLoRALayerState]
    rank: int
    alpha: float
    dropout: float

    def merge_all(self) -> int:
        """Merge all tracked `KerasLoRADense` layers back into their underlying `Dense` kernels.

        Restores all recorded container slots to their merged `Dense` layers.

        Returns:
            Count of merged adapter layers.

        """
        for state in self.layers.values():
            merged_dense = state.lora_layer.merge_weights()
            for seq, idx in state.container_slots:
                seq[idx] = merged_dense
        return len(self.layers)


type _WeakLoRAKey = KerasCausalLMProtocol | KerasLayerProtocol | _LayersContainerProtocol | _BackboneModelProtocol
_KERAS_LORA_REGISTRY: weakref.WeakKeyDictionary[_WeakLoRAKey, KerasLoRAAdapter] = weakref.WeakKeyDictionary()
_KERAS_LORA_ELEMENT_REGISTRY: weakref.WeakKeyDictionary[KerasLayerProtocol, KerasLoRAAdapter] = (
    weakref.WeakKeyDictionary()
)


def _register_lora_adapter_weakly(model: KerasPEFTTarget, adapter: KerasLoRAAdapter) -> None:
    """Register `adapter` in `WeakKeyDictionary` without retaining strong references or `id()` keys.

    For non-weakrefable built-in `list`/`tuple` containers, attaches weak entries keyed by the
    weakrefable `KerasLayerProtocol` elements inside the sequence so garbage collection is never blocked.

    Args:
        model: Target Keras model or sequence of layers.
        adapter: Constructed `KerasLoRAAdapter` metadata.

    """
    if isinstance(model, (list, tuple)):
        for item in model:
            if isinstance(item, KerasLayerProtocol):
                _KERAS_LORA_ELEMENT_REGISTRY[item] = adapter
        return
    assert isinstance(
        model, (KerasCausalLMProtocol, KerasLayerProtocol, _LayersContainerProtocol, _BackboneModelProtocol)
    )
    _KERAS_LORA_REGISTRY[model] = adapter


def get_keras_lora_adapter(
    model: KerasPEFTTarget,
) -> KerasLoRAAdapter | None:
    """Look up the registered `KerasLoRAAdapter` for a model instance using weak references only.

    Args:
        model: Target Keras model or layer container.

    Returns:
        Registered `KerasLoRAAdapter` or `None` if no LoRA adapters were injected.

    """
    if isinstance(model, (list, tuple)):
        for item in model:
            if isinstance(item, KerasLayerProtocol):
                elem_found = _KERAS_LORA_ELEMENT_REGISTRY.get(item)
                if elem_found is not None:
                    return elem_found
        return None
    assert isinstance(
        model, (KerasCausalLMProtocol, KerasLayerProtocol, _LayersContainerProtocol, _BackboneModelProtocol)
    )
    return _KERAS_LORA_REGISTRY.get(model)


def _collect_dense_layers_with_slots(
    model: KerasPEFTTarget,
) -> list[tuple[KerasDenseLayerProtocol, list[tuple[_LayerSlotSequence, int]]]]:
    """Collect all `keras.layers.Dense` layers and their mutable container index slots from `model`.

    Args:
        model: Keras model or layer container exposing `.layers`.

    Returns:
        Deduplicated list of `(dense_layer, mutable_container_slots)` pairs.

    """
    seen_layers: list[tuple[KerasDenseLayerProtocol, list[tuple[_LayerSlotSequence, int]]]] = []

    def _find_or_create_slots(dense: KerasDenseLayerProtocol) -> list[tuple[_LayerSlotSequence, int]]:
        """Return existing slot list for `dense` (by identity `is`) or append a new entry."""
        for existing_dense, existing_slots in seen_layers:
            if existing_dense is dense:
                return existing_slots
        fresh_slots: list[tuple[_LayerSlotSequence, int]] = []
        seen_layers.append((dense, fresh_slots))
        return fresh_slots

    def _visit_seq(seq: Sequence[KerasLayerProtocol | _LayersContainerProtocol]) -> None:
        """Visit a sequence of Keras layers and record `Dense` instances and mutable slots."""
        assert keras is not None
        dense_cls = keras.layers.Dense
        mutable_seq: _LayerSlotSequence | None = seq if isinstance(seq, _LayerSlotSequence) else None
        for idx, item in enumerate(seq):
            if isinstance(item, dense_cls) and isinstance(item, KerasDenseLayerProtocol):
                slot_list = _find_or_create_slots(item)
                if mutable_seq is not None:
                    slot_list.append((mutable_seq, idx))
            if isinstance(item, _LayersContainerProtocol):
                _visit_seq(item.layers)

    if isinstance(model, (list, tuple)):
        _visit_seq(model)
    elif isinstance(model, _LayersContainerProtocol):
        _visit_seq(model.layers)
    return seen_layers


def inject_lora[ModelT: KerasPEFTTarget](
    model: ModelT,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> tuple[ModelT, int]:
    """Inject `KerasLoRADense` adapter wrappers into target `Dense` layers of a Keras model.

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

    dense_entries = _collect_dense_layers_with_slots(model)
    adapted_states: dict[str, KerasLoRALayerState] = {}

    for dense, slots in dense_entries:
        layer_name = str(dense.name)
        if matches_target_module(layer_name, target_modules):
            lora_layer = KerasLoRADense(
                dense=dense,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            recorded_slots: list[tuple[_LayerSlotSequence, int]] = []
            for seq, idx in slots:
                seq[idx] = lora_layer
                recorded_slots.append((seq, idx))
            adapted_states[layer_name] = KerasLoRALayerState(
                layer_name=layer_name,
                base_layer=dense,
                lora_layer=lora_layer,
                container_slots=tuple(recorded_slots),
            )
        else:
            dense.trainable = False

    adapter = KerasLoRAAdapter(
        layers=adapted_states,
        rank=lora_r,
        alpha=float(lora_alpha),
        dropout=lora_dropout,
    )
    _register_lora_adapter_weakly(model, adapter)
    if not isinstance(model, (list, tuple)):
        register_quantized_artifact(
            model,
            method="lora",
            scales={name: state.lora_layer.scale for name, state in adapted_states.items()},
        )

    return model, len(adapted_states)


def merge_lora_weights[ModelT: KerasPEFTTarget](model: ModelT) -> ModelT:
    """Fold all tracked `KerasLoRADense` adapter weights back into base `Dense` kernels on `model`.

    Args:
        model: Keras model registered with `KerasLoRAAdapter` or containing `KerasLoRADense` layers.

    Returns:
        The model with all LoRA layers folded back into native `Dense` kernels.

    """
    adapter = get_keras_lora_adapter(model)
    if adapter is not None:
        adapter.merge_all()
        return model

    if isinstance(model, _LayersContainerProtocol):
        layers_seq = model.layers
        mutable_seq: _LayerSlotSequence | None = layers_seq if isinstance(layers_seq, _LayerSlotSequence) else None
        for idx, layer in enumerate(layers_seq):
            if isinstance(layer, KerasLoRADense):
                merged = layer.merge_weights()
                if mutable_seq is not None:
                    mutable_seq[idx] = merged
    return model


def count_parameters(model: KerasLayerProtocol | KerasCausalLMProtocol) -> tuple[int, int]:
    """Count total and trainable weights in a Keras model.

    Args:
        model: Keras model or layer exposing `.weights` and `.trainable_weights`.

    Returns:
        A tuple of `(total_variable_weights_count, trainable_weights_count)`.

    """
    total = (
        sum(math.prod(int(d) for d in w.shape) for w in model.weights) if isinstance(model, KerasLayerProtocol) else 0
    )
    trainable = (
        sum(math.prod(int(d) for d in w.shape) for w in model.trainable_weights)
        if isinstance(model, KerasLayerProtocol)
        else 0
    )
    return total, trainable


def apply_lora(
    model_name: str,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    **kwargs: BackendKwargValue
    | KerasCausalLMProtocol
    | KerasLayerProtocol
    | _BackboneModelProtocol
    | _LayersContainerProtocol,
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
        """Load or reuse Keras model and inject or merge LoRA weights.

        Returns:
            Count of injected or enabled LoRA target layers.

        Raises:
            ImportError: If `keras_nlp.models.GemmaCausalLM` is unavailable when loading a preset.

        """
        provided_model = kwargs.get("model")
        if isinstance(
            provided_model,
            (KerasCausalLMProtocol, KerasLayerProtocol, _BackboneModelProtocol, _LayersContainerProtocol),
        ):
            model: KerasPEFTTarget = provided_model
        else:
            gemma_causal_lm_cls = import_optional_symbol(
                "keras_nlp.models",
                "GemmaCausalLM",
                _GemmaCausalLMLoaderProtocol,
            )
            if gemma_causal_lm_cls is None:
                raise ImportError("keras_nlp.models.GemmaCausalLM is not available.")
            model = gemma_causal_lm_cls.from_preset(model_name)

        backbone: KerasBackboneProtocol | None = model.backbone if isinstance(model, _BackboneModelProtocol) else None

        if backbone is not None:
            backbone.enable_lora(rank=lora_r)
            matched_bb = 0
            for layer in backbone.layers:
                if matches_target_module(str(layer.name), target_modules):
                    matched_bb += 1
                layer.trainable = bool(layer.trainable_variables)
            injected_count = matched_bb if matched_bb > 0 else len(target_modules)
            logger.info("Enabled Keras native LoRA with rank %d", lora_r)
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

    from gemma_4_sql.backends.keras._env import resolve_and_validate_keras_backend

    backend_alias = str(kwargs.get("backend_alias", "keras"))
    explicit_kb = (
        str(kwargs["keras_backend"]) if "keras_backend" in kwargs and kwargs["keras_backend"] is not None else None
    )
    active_kb = resolve_and_validate_keras_backend(backend_alias, explicit_kb)
    wrapper_kwargs: dict[str, StatusValue] = {
        k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None
    }
    res = apply_lora_wrapper(
        backend_alias,
        model_name,
        target_modules,
        lora_r,
        lora_alpha,
        lora_dropout,
        _inject,
        **wrapper_kwargs,
    )
    res["keras_backend"] = active_kb
    return res


__all__ = [
    "KerasDenseLayerProtocol",
    "KerasLoRAAdapter",
    "KerasLoRADense",
    "KerasLoRALayerState",
    "_KerasTensorInputProtocol",
    "apply_lora",
    "count_parameters",
    "get_keras_lora_adapter",
    "inject_lora",
    "merge_lora_weights",
]
