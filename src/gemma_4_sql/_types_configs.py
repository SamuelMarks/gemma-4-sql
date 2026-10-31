"""Configuration dataclasses and quantized artifact registry for gemma-4-sql."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar, Union

from gemma_4_sql._types_json import JSONValue
from gemma_4_sql._types_models import (
    ForwardOutput,
    HFTokenizerProtocol,
    ModelType,
    TokenizerProtocol,
)
from gemma_4_sql._types_results import DataLoaderProtocol
from gemma_4_sql._types_tensors import TensorType

TQuantModel = TypeVar("TQuantModel")


def _empty_scales_dict() -> dict[str, float]:
    """Return an empty quantization scale dictionary."""
    return {}


@dataclass
class QuantizedModelArtifact(Generic[TQuantModel]):
    """Strongly-typed wrapper recording quantization state and scales for a model."""

    model: TQuantModel
    method: str
    is_quantized: bool = True
    scales: dict[str, float] = field(default_factory=_empty_scales_dict)


@dataclass
class _QuantizedMetadata:
    """Internal metadata record stored in `_QUANTIZED_ARTIFACT_WEAK_REFS` without a strong model cycle."""

    method: str
    is_quantized: bool
    scales: dict[str, float]


_QUANTIZED_ARTIFACT_WEAK_REFS: dict[int, _QuantizedMetadata] = {}
_QUANTIZED_ARTIFACT_FALLBACK_REGISTRY: dict[int, _QuantizedMetadata] = {}


def clear_quantized_artifact_registry() -> None:
    """Clear all cached entries from the quantized model artifact registry."""
    _QUANTIZED_ARTIFACT_WEAK_REFS.clear()
    _QUANTIZED_ARTIFACT_FALLBACK_REGISTRY.clear()


def register_quantized_artifact(
    model: TQuantModel,
    method: str,
    scales: dict[str, float] | None = None,
) -> QuantizedModelArtifact[TQuantModel]:
    """Register quantization metadata for a model in the memory-safe artifact registry.

    Args:
        model: Target model instance.
        method: Applied quantization method identifier ('int8', 'int4', 'awq', etc.).
        scales: Optional per-layer or per-parameter quantization scale factors.

    Returns:
        Registered `QuantizedModelArtifact` wrapping `model`.

    """
    resolved_scales = dict(scales) if scales is not None else {}
    meta = _QuantizedMetadata(
        method=method,
        is_quantized=True,
        scales=resolved_scales,
    )
    model_id = id(model)

    def _on_collect() -> None:
        """Remove `model_id` from `_QUANTIZED_ARTIFACT_WEAK_REFS` upon garbage collection."""
        _QUANTIZED_ARTIFACT_WEAK_REFS.pop(model_id, None)

    try:
        weakref.finalize(model, _on_collect)
        _QUANTIZED_ARTIFACT_WEAK_REFS[model_id] = meta
    except TypeError:
        _QUANTIZED_ARTIFACT_FALLBACK_REGISTRY[model_id] = meta
    return QuantizedModelArtifact(
        model=model,
        method=method,
        is_quantized=True,
        scales=resolved_scales,
    )


def get_quantized_artifact(model: TQuantModel) -> QuantizedModelArtifact[TQuantModel] | None:
    """Retrieve the registered `QuantizedModelArtifact` for a quantized model instance.

    Args:
        model: Target model instance to inspect.

    Returns:
        Registered `QuantizedModelArtifact` if the model was quantized, or `None`.

    """
    model_id = id(model)
    meta = _QUANTIZED_ARTIFACT_WEAK_REFS.get(model_id) or _QUANTIZED_ARTIFACT_FALLBACK_REGISTRY.get(model_id)
    if meta is not None:
        return QuantizedModelArtifact(
            model=model,
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
type BackendKwargValue = JSONValue | Path | ModelType | TensorType | TokenizerProtocol | HFTokenizerProtocol | DataLoaderProtocol[Mapping[str, TensorType | Sequence[int]]] | Iterable[Mapping[str, TensorType | Sequence[int]]] | Sequence[TensorType] | Mapping[str, TensorType]
ExtraKwargsDict = Mapping[str, ExtraValue]


def _empty_extra_kwargs() -> dict[str, ExtraValue]:
    """Return an empty dictionary for configuration extra_kwargs."""
    return {}


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization (DPO) execution."""

    model_name: str
    dataset: str
    beta: float = 0.1
    epochs: int = 1
    learning_rate: float = 1e-05
    batch_size: int = 2
    raise_on_error: bool = False
    extra_kwargs: ExtraKwargsDict = field(default_factory=_empty_extra_kwargs)


@dataclass
class ETLConfig:
    """Configuration for ETL dataset loading and tokenization pipelines."""

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
    stage: str = "sft"
    raise_on_error: bool = False


@dataclass
class TrainingConfig:
    """Configuration for pretraining, SFT, and post-training execution."""

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
