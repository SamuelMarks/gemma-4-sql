"""Configuration dataclasses and memory-safe quantized artifact registry for gemma-4-sql."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Protocol, TypeVar, Union, final, runtime_checkable

from gemma_4_sql._types_json import JSONValue
from gemma_4_sql._types_models import (
    ForwardOutput,
    HFTokenizerProtocol,
    KerasBackboneProtocol,
    KerasLayerProtocol,
    KerasVariableProtocol,
    MlxCausalLMProtocol,
    ModelConfigProtocol,
    ModelType,
    PyTreeNode,
    QuantizableModelType,
    TokenizerProtocol,
    TrainableModelType,
)
from gemma_4_sql._types_results import DataLoaderProtocol
from gemma_4_sql._types_tensors import TensorType
from gemma_4_sql.exceptions import ConfigurationError


@runtime_checkable
class _QuantConfigAttrProtocol(Protocol):
    """Structural protocol for configuration objects exposing `quantization_config`."""

    quantization_config: JSONValue | None


@runtime_checkable
class _ConfigTargetProtocol(Protocol):
    """Structural protocol for models exposing a `.config` descriptor."""

    @property
    def config(self) -> _QuantConfigAttrProtocol | ModelConfigProtocol | Hashable | Mapping[str, JSONValue] | None:
        """Provide model configuration object."""
        ...


@runtime_checkable
class _ShapeTargetProtocol(Protocol):
    """Structural protocol for tensor-like objects exposing `.shape`."""

    @property
    def shape(self) -> Sequence[int | None] | None:
        """Provide tensor dimensions."""
        ...


@runtime_checkable
class _WeightsTargetProtocol(Protocol):
    """Structural protocol for layers or models exposing `.weights`."""

    @property
    def weights(self) -> Sequence[KerasVariableProtocol | TensorType | _ShapeTargetProtocol | str]:
        """Provide weight variables."""
        ...


@runtime_checkable
class _KernelTargetProtocol(Protocol):
    """Structural protocol for Keras `Dense` layers exposing `.kernel`."""

    @property
    def kernel(self) -> KerasVariableProtocol | TensorType:
        """Provide layer kernel variable."""
        ...


@runtime_checkable
class _WeightAttrTargetProtocol(Protocol):
    """Structural protocol for linear layers exposing `.weight`."""

    @property
    def weight(self) -> TensorType | Sequence[float]:
        """Provide layer weight tensor."""
        ...


@runtime_checkable
class _LayersAttrTargetProtocol(Protocol):
    """Structural protocol for models exposing `.layers`."""

    @property
    def layers(self) -> Sequence[KerasLayerProtocol | _WeightsTargetProtocol | _LayersAttrTargetProtocol]:
        """Provide child layers."""
        ...


@runtime_checkable
class _BackboneAttrTargetProtocol(Protocol):
    """Structural protocol for models exposing `.backbone`."""

    @property
    def backbone(self) -> KerasBackboneProtocol | _LayersAttrTargetProtocol:
        """Provide underlying model backbone."""
        ...


@runtime_checkable
class _DataParamTargetProtocol(Protocol):
    """Structural protocol for parameter wrappers exposing `.data`."""

    @property
    def data(self) -> TensorType:
        """Provide underlying parameter tensor."""
        ...


@runtime_checkable
class _NamedParamsTargetProtocol(Protocol):
    """Structural protocol for PyTorch-like modules exposing `.named_parameters()`."""

    def named_parameters(self) -> Iterable[tuple[str, TensorType | _DataParamTargetProtocol]]:
        """Yield `(name, parameter)` pairs."""
        ...


@runtime_checkable
class _NamedModulesTargetProtocol(Protocol):
    """Structural protocol for MLX/PyTorch modules exposing `.named_modules()`."""

    def named_modules(self) -> Iterable[tuple[str, _WeightAttrTargetProtocol | ModelType | KerasLayerProtocol]]:
        """Yield `(name, submodule)` pairs."""
        ...


@runtime_checkable
class _SavableTargetProtocol(Protocol):
    """Structural protocol for models supporting `.save(filepath)`."""

    def save(self, filepath: str, /) -> None:
        """Persist model artifact to `filepath`."""
        ...


@runtime_checkable
class _NativeQuantizeTargetProtocol(Protocol):
    """Structural protocol for models supporting `.quantize(mode)`."""

    def quantize(self, mode: str, /) -> None:
        """Quantize model weights in place."""
        ...


type QuantizableTargetBound = (
    Hashable
    | BackendKwargValue
    | ModelType
    | _ConfigTargetProtocol
    | QuantizableModelType
    | TrainableModelType
    | KerasLayerProtocol
    | _WeightsTargetProtocol
    | _KernelTargetProtocol
    | _WeightAttrTargetProtocol
    | _LayersAttrTargetProtocol
    | _BackboneAttrTargetProtocol
    | _NamedParamsTargetProtocol
    | _NamedModulesTargetProtocol
    | _SavableTargetProtocol
    | _NativeQuantizeTargetProtocol
    | Mapping[str, PyTreeNode]
    | Mapping[str, JSONValue]
    | Mapping[str, QuantizableTargetBound]
    | Sequence[QuantizableTargetBound]
    | Callable[..., ForwardOutput]
)

TQuantModel = TypeVar("TQuantModel", bound=QuantizableTargetBound)


def _empty_scales_dict() -> dict[str, float]:
    """Return an empty quantization scale dictionary.

    Returns:
        Empty dictionary mapping layer or parameter names to float scales.

    """
    return {}


@final
@dataclass
class QuantizedModelArtifact(Generic[TQuantModel]):
    """Strongly-typed wrapper recording quantization state and scales for a model.

    Explicitly wraps both weakrefable models and non-weakrefable structures (such as
    MaxText parameter PyTree mappings) via composition without relying on `id()` side-tables.

    Attributes:
        model: Underlying model instance or parameter PyTree wrapped by the artifact.
        method: Applied quantization method identifier (`'int8'`, `'int4'`, `'awq'`, `'gptq'`, etc.).
        is_quantized: Boolean flag indicating whether quantization has been applied.
        scales: Per-layer or per-parameter floating-point quantization scale factors.

    """

    model: TQuantModel
    method: str
    is_quantized: bool = True
    scales: dict[str, float] = field(default_factory=_empty_scales_dict)


@dataclass
class _QuantizedMetadata:
    """Internal metadata record stored in weak registries without a strong model reference cycle.

    Attributes:
        method: Quantization method identifier.
        is_quantized: Whether the model is quantized.
        scales: Quantization scale factors per layer or tensor.

    """

    method: str
    is_quantized: bool
    scales: dict[str, float]


_QUANTIZED_ARTIFACT_REGISTRY: weakref.WeakKeyDictionary[Hashable, _QuantizedMetadata] = weakref.WeakKeyDictionary()
_UNHASHABLE_WEAK_REFS: list[tuple[weakref.ref[MlxCausalLMProtocol], _QuantizedMetadata]] = []


def clear_quantized_artifact_registry() -> None:
    """Clear all cached entries from the quantized model artifact weak registries."""
    _QUANTIZED_ARTIFACT_REGISTRY.clear()
    _UNHASHABLE_WEAK_REFS.clear()


def _prune_dead_unhashable_refs(_dead_ref: weakref.ref[MlxCausalLMProtocol] | None = None) -> None:
    """Remove garbage-collected weak references from `_UNHASHABLE_WEAK_REFS`.

    Args:
        _dead_ref: Optional finalized weak reference supplied by `weakref.ref` callback.

    """
    _UNHASHABLE_WEAK_REFS[:] = [(ref, meta) for ref, meta in _UNHASHABLE_WEAK_REFS if ref() is not None]


def register_quantized_artifact(
    model: TQuantModel,
    method: str,
    scales: dict[str, float] | None = None,
) -> QuantizedModelArtifact[TQuantModel]:
    """Register quantization metadata in `WeakKeyDictionary` when weakrefable and return a `QuantizedModelArtifact`.

    Non-weakrefable structures (such as `dict` PyTrees or `__slots__` classes without `__weakref__`)
    are never retained in global registries; callers compose them directly via the returned
    `QuantizedModelArtifact` wrapper.

    Args:
        model: Target model instance or parameter structure.
        method: Applied quantization method identifier (`'int8'`, `'int4'`, `'awq'`, `'gptq'`, etc.).
        scales: Optional per-layer or per-parameter quantization scale factors.

    Returns:
        `QuantizedModelArtifact` wrapping `model` and its quantization metadata.

    """
    orig_model: TQuantModel = model
    resolved_scales = dict(scales) if scales is not None else {}
    meta = _QuantizedMetadata(
        method=method,
        is_quantized=True,
        scales=resolved_scales,
    )
    if isinstance(model, MlxCausalLMProtocol):
        try:
            model_ref: weakref.ref[MlxCausalLMProtocol] = weakref.ref(
                model,
                lambda _r: _prune_dead_unhashable_refs(),
            )
            _UNHASHABLE_WEAK_REFS[:] = [
                (ref, existing) for ref, existing in _UNHASHABLE_WEAK_REFS if ref() is not None and ref() is not model
            ]
            _UNHASHABLE_WEAK_REFS.append((model_ref, meta))
        except TypeError:
            pass
    elif not isinstance(model, (Mapping, list, dict, set)):
        try:
            if isinstance(model, Hashable):
                _QUANTIZED_ARTIFACT_REGISTRY[model] = meta
        except TypeError:
            # Non-weakrefable objects rely on the returned QuantizedModelArtifact wrapper.
            pass
    return QuantizedModelArtifact(
        model=orig_model,
        method=method,
        is_quantized=True,
        scales=resolved_scales,
    )


def get_quantized_artifact(
    model: TQuantModel | QuantizedModelArtifact[TQuantModel],
) -> QuantizedModelArtifact[TQuantModel] | None:
    """Retrieve the `QuantizedModelArtifact` for a weakrefable model or unwrap an explicit `QuantizedModelArtifact`.

    Args:
        model: Target model instance or `QuantizedModelArtifact` wrapper to inspect.

    Returns:
        `QuantizedModelArtifact` if `model` is a wrapper or registered in the weak registry, else `None`.

    """
    if isinstance(model, QuantizedModelArtifact):
        return QuantizedModelArtifact(
            model=model.model,
            method=model.method,
            is_quantized=model.is_quantized,
            scales=dict(model.scales),
        )
    unwrapped_model: TQuantModel = model
    meta: _QuantizedMetadata | None = None
    if isinstance(model, MlxCausalLMProtocol):
        for ref, candidate_meta in _UNHASHABLE_WEAK_REFS:
            if ref() is model:
                meta = candidate_meta
                break
    elif not isinstance(model, (Mapping, list, dict, set)):
        try:
            if isinstance(model, Hashable):
                meta = _QUANTIZED_ARTIFACT_REGISTRY.get(model)
        except TypeError:
            meta = None
    if meta is not None:
        return QuantizedModelArtifact(
            model=unwrapped_model,
            method=meta.method,
            is_quantized=meta.is_quantized,
            scales=dict(meta.scales),
        )
    return None


ExtraValue = Union[
    JSONValue,
    Path,
    ModelType,
    TensorType,
    TokenizerProtocol,
    HFTokenizerProtocol,
    DataLoaderProtocol[Mapping[str, TensorType | Sequence[int]]],
    Iterable[Mapping[str, TensorType | Sequence[int]]],
    Callable[..., ForwardOutput | JSONValue],
]
type BackendKwargValue = (
    JSONValue
    | Path
    | ModelType
    | TensorType
    | TokenizerProtocol
    | HFTokenizerProtocol
    | DataLoaderProtocol[Mapping[str, TensorType | Sequence[int]]]
    | Iterable[Mapping[str, TensorType | Sequence[int]]]
    | Sequence[TensorType]
    | Mapping[str, TensorType]
)
ExtraKwargsDict = Mapping[str, ExtraValue]


def _empty_extra_kwargs() -> dict[str, ExtraValue]:
    """Return an empty dictionary for configuration `extra_kwargs`.

    Returns:
        Empty mutable dictionary of extra keyword arguments.

    """
    return {}


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization (DPO) execution.

    Attributes:
        model_name: Target policy/reference model name or checkpoint path.
        dataset: Preference dataset identifier or local path.
        beta: Positive KL-divergence regularization strength (`> 0.0`).
        epochs: Non-negative number of DPO training epochs (`>= 0`).
        learning_rate: Positive optimizer learning rate (`> 0.0`).
        batch_size: Positive per-step batch size (`> 0`).
        raise_on_error: Whether operational errors should raise `TrainingError` immediately.
        extra_kwargs: Additional backend-specific keyword options.

    """

    model_name: str
    dataset: str
    beta: float = 0.1
    epochs: int = 1
    learning_rate: float = 1e-05
    batch_size: int = 2
    raise_on_error: bool = False
    extra_kwargs: ExtraKwargsDict = field(default_factory=_empty_extra_kwargs)

    def __post_init__(self) -> None:
        """Validate DPO hyperparameter invariants.

        Raises:
            ConfigurationError: If `beta <= 0.0`, `epochs < 0`, `learning_rate <= 0.0`, or `batch_size <= 0`.

        """
        if self.beta <= 0.0:
            raise ConfigurationError(f"beta must be positive, got {self.beta}")
        if self.epochs < 0:
            raise ConfigurationError(f"epochs must be non-negative, got {self.epochs}")
        if self.learning_rate <= 0.0:
            raise ConfigurationError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ConfigurationError(f"batch_size must be positive, got {self.batch_size}")


@dataclass
class ETLConfig:
    """Configuration for ETL dataset loading and tokenization pipelines.

    Attributes:
        dataset_name: Hugging Face dataset name or local dataset identifier.
        split: Dataset split name (e.g., `'train'`, `'validation'`, `'test'`).
        batch_size: Positive batch size for DataLoader construction (`> 0`).
        distributed: Whether distributed sharding/sampling is enabled.
        tokenizer_name: Optional Hugging Face tokenizer identifier.
        duckdb_path: Optional DuckDB file path for analytical SQL dataset loading.
        duckdb_table: Optional DuckDB table or view name to query.
        modality: Input modality (`'text'`, `'image'`, `'audio'`, `'multimodal'`).
        image_column: Optional dataset column name containing image payloads.
        audio_column: Optional dataset column name containing audio payloads.
        image_path: Optional default image file path for multimodal ETL.
        audio_path: Optional default audio file path for multimodal ETL.
        stage: Training stage (`'pretrain'`, `'sft'`, or `'posttrain'`).
        raise_on_error: Whether operational errors should raise `ETLError` immediately.

    """

    dataset_name: str
    split: str
    batch_size: int = 32
    distributed: bool = False
    tokenizer_name: str | None = None
    duckdb_path: str | None = None
    duckdb_table: str | None = None
    modality: str = "text"
    image_column: str | None = None
    audio_column: str | None = None
    image_path: str | None = None
    audio_path: str | None = None
    stage: str = "sft"
    raise_on_error: bool = False

    def __post_init__(self) -> None:
        """Validate ETL configuration invariants.

        Raises:
            ConfigurationError: If `batch_size <= 0`.

        """
        if self.batch_size <= 0:
            raise ConfigurationError(f"batch_size must be positive, got {self.batch_size}")


@dataclass
class TrainingConfig:
    """Configuration for pretraining, SFT, and post-training execution.

    Attributes:
        action: Training stage action (`'pretrain'`, `'sft'`, `'posttrain'`, or `''`).
        model_name: Target model preset name or checkpoint path.
        dataset: Training dataset identifier or path.
        epochs: Non-negative number of training epochs (`>= 0`).
        learning_rate: Positive optimizer learning rate (`> 0.0`).
        batch_size: Positive per-step batch size (`> 0`).
        backend: Backend engine identifier (`'jax'`, `'pytorch'`, `'keras'`, `'maxtext'`, `'mlx'`).
        distributed_strategy: Distributed strategy identifier (`'none'`, `'ddp'`, `'fsdp'`, etc.).
        modality: Input modality (`'text'`, `'image'`, `'audio'`, `'multimodal'`).
        raise_on_error: Whether operational errors should raise `TrainingError` immediately.
        extra_kwargs: Additional backend-specific keyword options.

    """

    action: str = ""
    model_name: str = "gemma-4"
    dataset: str = "dummy"
    epochs: int = 1
    learning_rate: float = 0.0001
    batch_size: int = 2
    backend: str = "jax"
    distributed_strategy: str = "none"
    modality: str = "text"
    raise_on_error: bool = False
    extra_kwargs: ExtraKwargsDict = field(default_factory=_empty_extra_kwargs)

    def __post_init__(self) -> None:
        """Validate training hyperparameter invariants.

        Raises:
            ConfigurationError: If `epochs < 0`, `learning_rate <= 0.0`, or `batch_size <= 0`.

        """
        if self.epochs < 0:
            raise ConfigurationError(f"epochs must be non-negative, got {self.epochs}")
        if self.learning_rate <= 0.0:
            raise ConfigurationError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ConfigurationError(f"batch_size must be positive, got {self.batch_size}")


__all__ = [
    "BackendKwargValue",
    "DPOConfig",
    "ETLConfig",
    "ExtraKwargsDict",
    "ExtraValue",
    "QuantizedModelArtifact",
    "TQuantModel",
    "TrainingConfig",
    "clear_quantized_artifact_registry",
    "get_quantized_artifact",
    "register_quantized_artifact",
]
