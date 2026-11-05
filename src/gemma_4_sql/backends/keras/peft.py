"""Keras-specific PEFT / LoRA implementation."""

from __future__ import annotations

import contextlib
import logging
import math
import weakref
from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeGuard, runtime_checkable

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
class _KerasOpsMatmulProtocol(Protocol):
    """Protocol for `keras.ops` providing `matmul` returning `TfTensorProtocol`."""

    def matmul(
        self,
        x1: TensorType | KerasVariableProtocol,
        x2: TensorType | KerasVariableProtocol,
        /,
    ) -> TfTensorProtocol:
        """Compute matrix multiplication of `x1` and `x2`."""
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
class _KerasTensorInputProtocol(Protocol):
    """Protocol for any Keras tensor or symbolic `OpenVINOKerasTensor` exposing `.shape`."""

    @property
    def shape(self) -> Sequence[int | None] | None:
        """Provide the tensor shape dimensions."""
        ...


class _LayersContainerProtocol(Protocol):
    """Protocol for any Keras model or container exposing `.layers`."""

    @property
    def layers(self) -> Sequence[KerasLayerProtocol | _LayersContainerProtocol]:
        """Provide the contained layers."""
        ...


type KerasPEFTTarget = KerasCausalLMProtocol | KerasLayerProtocol | KerasDenseLayerProtocol | _BackboneModelProtocol | _LayersContainerProtocol | Sequence[KerasLayerProtocol | _LayersContainerProtocol]


def _is_keras_tensor_input(val: TensorType | _KerasTensorInputProtocol) -> TypeGuard[TensorType]:
    """Narrow `val` to `TensorType` for `KerasLoRADense` shape and forward call.

    Args:
        val: Input tensor passed to `KerasLoRADense`.

    Returns:
        `True` if `val` is a tensor-like object with `.shape`.

    """
    return not isinstance(val, (str, int, float))


def _as_keras_input_t[InputT: TensorType | _KerasTensorInputProtocol](
    val: TfTensorProtocol,
    _sample: InputT,
) -> TypeGuard[InputT]:
    """Narrow `val` back to `InputT` for `KerasLoRADense.__call__`.

    Args:
        val: Output tensor from `KerasLoRADense.call`.
        _sample: Input tensor of type `InputT`.

    Returns:
        `True` if `val` is not a plain `str` or `int`.

    """
    _ = _sample
    return not isinstance(val, (str, int))


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

    def __call__[InputT: TensorType | _KerasTensorInputProtocol](self, inputs: InputT, training: bool | None = None) -> InputT:
        """Invoke the layer forward pass, building LoRA weights lazily if needed."""
        in_ref: TensorType | _KerasTensorInputProtocol = inputs
        assert _is_keras_tensor_input(in_ref)
        if not self._lora_built:
            self.build(tuple(in_ref.shape))
        out = self.call(in_ref, training=training)
        assert _as_keras_input_t(out, inputs)
        return out

    def call(self, inputs: TensorType, training: bool | None = None) -> TfTensorProtocol:
        """Execute the forward pass combining frozen base projection and low-rank adapter.

        Args:
            inputs: Input tensor.
            training: Boolean indicating training or inference mode for dropout.

        Returns:
            Output tensor of shape `(..., units)`.

        """
        base_out = self.dense(inputs)
        dropped: TensorType = self.dropout(inputs, training=training) if self.dropout is not None else inputs
        k_ops = _typed_ops()
        lora_out: TfTensorProtocol = k_ops.matmul(k_ops.matmul(dropped, self.lora_a), self.lora_b)
        scaled_lora: TfTensorProtocol = lora_out * self.scale
        res: TfTensorProtocol = base_out + scaled_lora
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

        k_ops = _typed_ops()
        raw_matmul: TfTensorProtocol = k_ops.matmul(self.lora_a, self.lora_b)
        delta: TfTensorProtocol = raw_matmul * self.scale
        self.dense.kernel.assign(self.dense.kernel + delta)
        self.dense.trainable = True
        return self.dense


type _LayerSlotSequence = MutableSequence[KerasLayerProtocol | _LayersContainerProtocol]


@dataclass(frozen=True)
class KerasLoRALayerState:
    """State record for a single Keras `Dense` layer adapted by `KerasLoRAAdapter`."""

    layer_name: str
    base_layer: KerasDenseLayerProtocol
    lora_layer: KerasLoRADense
    container_slots: tuple[tuple[_LayerSlotSequence, int], ...] = ()


@dataclass
class KerasLoRAAdapter:
    """Typed container tracking all `KerasLoRALayerState` entries injected into a Keras model."""

    layers: dict[str, KerasLoRALayerState]
    rank: int
    alpha: float
    dropout: float

    def merge_all(self) -> int:
        """Merge all tracked `KerasLoRADense` layers back into their underlying `Dense` kernels and restore container slots.

        Returns:
            Count of merged adapter layers.

        """
        for state in self.layers.values():
            merged_dense = state.lora_layer.merge_weights()
            for seq, idx in state.container_slots:
                seq[idx] = merged_dense
        return len(self.layers)


_KERAS_LORA_REGISTRY: weakref.WeakKeyDictionary[KerasCausalLMProtocol | KerasLayerProtocol, KerasLoRAAdapter] = weakref.WeakKeyDictionary()
_KERAS_LORA_ID_FALLBACK: dict[int, KerasLoRAAdapter] = {}


def _is_registry_key(model: KerasPEFTTarget) -> TypeGuard[KerasCausalLMProtocol | KerasLayerProtocol]:
    """Narrow `model` to a weakref-compatible `KerasCausalLMProtocol | KerasLayerProtocol` key.

    Args:
        model: Target Keras model or container.

    Returns:
        `True` if `model` is not a plain `list` or `tuple`.

    """
    return not isinstance(model, (list, tuple))


def _has_layers(obj: KerasPEFTTarget) -> TypeGuard[_LayersContainerProtocol]:
    """Narrow `obj` to `_LayersContainerProtocol` for `.layers` traversal.

    Args:
        obj: Target Keras model or layer.

    Returns:
        `True` if `obj` is not a plain `list` or `tuple`.

    """
    return not isinstance(obj, (list, tuple))


def _has_backbone(model: KerasPEFTTarget) -> TypeGuard[_BackboneModelProtocol]:
    """Narrow `model` to `_BackboneModelProtocol` for `.backbone` access.

    Args:
        model: Target Keras model.

    Returns:
        `True` if `model` conforms to `_BackboneModelProtocol`.

    """
    return isinstance(model, _BackboneModelProtocol)


def get_keras_lora_adapter(
    model: KerasPEFTTarget,
) -> KerasLoRAAdapter | None:
    """Look up the registered `KerasLoRAAdapter` for a model instance.

    Args:
        model: Target Keras model or layer container.

    Returns:
        Registered `KerasLoRAAdapter` or `None` if no LoRA adapters were injected.

    """
    with contextlib.suppress(TypeError, AssertionError):
        assert _is_registry_key(model)
        found = _KERAS_LORA_REGISTRY.get(model)
        if found is not None:
            return found
    return _KERAS_LORA_ID_FALLBACK.get(id(model))


def _is_mutable_layer_seq(
    seq: Sequence[KerasLayerProtocol | _LayersContainerProtocol],
) -> TypeGuard[_LayerSlotSequence]:
    """Check whether `seq` is a mutable sequence supporting item assignment."""
    return isinstance(seq, MutableSequence)


def _collect_dense_layers_with_slots(
    model: KerasPEFTTarget,
) -> list[tuple[KerasDenseLayerProtocol, list[tuple[_LayerSlotSequence, int]]]]:
    """Collect all `keras.layers.Dense` layers and their mutable container index slots from `model`.

    Args:
        model: Keras model or layer container exposing `.layers`.

    Returns:
        Deduplicated list of `(dense_layer, mutable_container_slots)` pairs.

    """
    seen: dict[int, tuple[KerasDenseLayerProtocol, list[tuple[_LayerSlotSequence, int]]]] = {}

    def _visit_seq(seq: Sequence[KerasLayerProtocol | _LayersContainerProtocol]) -> None:
        """Visit a sequence of Keras layers and record `Dense` instances and mutable slots."""
        mutable_seq = seq if _is_mutable_layer_seq(seq) else None
        for idx, item in enumerate(seq):
            item_id = id(item)
            if isinstance(item, keras.layers.Dense):
                if item_id not in seen:
                    seen[item_id] = (item, [])
                if mutable_seq is not None:
                    seen[item_id][1].append((mutable_seq, idx))
            with contextlib.suppress(AttributeError, TypeError):
                assert _has_layers(item)
                _visit_seq(item.layers)

    if isinstance(model, (list, tuple)):
        _visit_seq(model)
    else:
        with contextlib.suppress(AttributeError, TypeError):
            assert _has_layers(model)
            _visit_seq(model.layers)
    return list(seen.values())


def inject_lora[ModelT: KerasPEFTTarget](
    model: ModelT,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> tuple[ModelT, int]:
    """Inject `KerasLoRADense` adapter wrappers into target `Dense` layers of a Keras model.

    Replaces matching `Dense` layers in-place inside mutable `.layers` containers so that
    forward passes through `model` route directly through `KerasLoRADense`, and tracks adapted
    layers in a typed `KerasLoRAAdapter` (`dict[str, KerasLoRALayerState]`).

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
                with contextlib.suppress(TypeError, ValueError, AttributeError):
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
    model_id = id(model)
    with contextlib.suppress(TypeError, AssertionError):
        assert _is_registry_key(model)
        _KERAS_LORA_REGISTRY[model] = adapter
    _KERAS_LORA_ID_FALLBACK[model_id] = adapter
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

    with contextlib.suppress(AttributeError, TypeError):
        assert _has_layers(model)
        layers_seq = model.layers
        mutable_seq = layers_seq if _is_mutable_layer_seq(layers_seq) else None
        for idx, layer in ((i, l) for i, l in enumerate(layers_seq) if isinstance(l, KerasLoRADense)):
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
    total = 0
    trainable = 0
    with contextlib.suppress(AttributeError, TypeError):
        total = sum(math.prod(int(d) for d in w.shape) for w in model.weights)
    with contextlib.suppress(AttributeError, TypeError):
        trainable = sum(math.prod(int(d) for d in w.shape) for w in model.trainable_weights)
    return total, trainable


def _is_keras_peft_target(
    val: BackendKwargValue | KerasPEFTTarget,
) -> TypeGuard[KerasPEFTTarget]:
    """Check whether `val` is a valid `KerasPEFTTarget` model or layer container."""
    return val is not None and not isinstance(val, (str, int, float, bool))


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
        if provided_model is not None and _is_keras_peft_target(provided_model):
            model: KerasPEFTTarget = provided_model
        else:
            gemma_causal_lm_cls = optional_attr(
                "keras_nlp.models",
                "GemmaCausalLM",
                _GemmaCausalLMLoaderProtocol,
            )
            if gemma_causal_lm_cls is None:
                raise ImportError("keras_nlp.models.GemmaCausalLM is not available.")
            model = gemma_causal_lm_cls.from_preset(model_name)

        backbone: KerasBackboneProtocol | None = model.backbone if _has_backbone(model) else None

        if backbone is not None:
            backbone.enable_lora(rank=lora_r)
            with contextlib.suppress(AttributeError, TypeError):
                for layer in backbone.layers:
                    with contextlib.suppress(AttributeError):
                        layer.trainable = bool(layer.trainable_variables)
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
