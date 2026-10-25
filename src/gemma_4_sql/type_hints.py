"""Strongly-typed protocols, TypedDicts, and configuration dataclasses for gemma-4-sql."""

from __future__ import annotations

import types
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, NotRequired, Protocol, Self, TypedDict, TypeVar, Union, overload, runtime_checkable

__all__ = [
    "AgentAttemptDict",
    "AgentLoopResultDict",
    "AgentResultDict",
    "AudioInput",
    "BenchmarkResultDict",
    "BenchmarkSubmoduleProtocol",
    "CausalLMProtocol",
    "ChatMessageDict",
    "ChatTurnResultDict",
    "CursorProtocol",
    "DPOConfig",
    "DPOLoopState",
    "DPOResultDict",
    "DPOSubmoduleProtocol",
    "DataLoaderProtocol",
    "DatabaseConnectionProtocol",
    "DataloaderResultDict",
    "ETLConfig",
    "ETLResultDict",
    "ETLSubmoduleProtocol",
    "EmbedderModelProtocol",
    "EvaluationMetricsDict",
    "EvaluationResultDict",
    "ExportResultDict",
    "ExportSubmoduleProtocol",
    "FastAPIAppProtocol",
    "FewShotPromptResultDict",
    "FewShotResultDict",
    "FlaxNNXModelProtocol",
    "ImageInput",
    "InferenceResultDict",
    "InferenceSubmoduleProtocol",
    "JSONDict",
    "JSONPrimitive",
    "JSONValue",
    "JaxCausalLMProtocol",
    "JaxModelProtocol",
    "KerasCausalLMProtocol",
    "KerasModelProtocol",
    "KerasTrainerState",
    "LatencyStatsDict",
    "LogMetricsResultDict",
    "LoggingSubmoduleProtocol",
    "MLXModelProtocol",
    "MaxTextCausalLMProtocol",
    "MaxTextModelProtocol",
    "MaxTextTrainerState",
    "MlxCausalLMProtocol",
    "MlxModelProtocol",
    "ModelType",
    "MultimodalInput",
    "NDArrayProtocol",
    "PEFTResultDict",
    "PEFTSubmoduleProtocol",
    "QuantizableModelProtocol",
    "QuantizableModelType",
    "QuantizeResultDict",
    "QuantizeSubmoduleProtocol",
    "QuantizedModelArtifact",
    "ServeResultDict",
    "ServeSubmoduleProtocol",
    "TensorIndex",
    "TensorProtocol",
    "TensorType",
    "TokenizerProtocol",
    "TorchCausalLMProtocol",
    "TorchModelProtocol",
    "TorchTrainerState",
    "TrainResultDict",
    "TrainSubmoduleProtocol",
    "TrainableModelProtocol",
    "TrainableModelType",
    "TrainerState",
    "TrainingConfig",
    "TrainingLoopState",
    "clear_quantized_artifact_registry",
    "get_quantized_artifact",
    "ndarray",
    "register_quantized_artifact",
]

JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Union[JSONPrimitive, Sequence["JSONValue"], Mapping[str, "JSONValue"]]
JSONDict = dict[str, Any]

TensorIndex = Union[int, slice, types.EllipsisType, tuple[Union[int, slice, types.EllipsisType], ...]]


@runtime_checkable
class NDArrayProtocol(Protocol):
    """Protocol representing a NumPy-compatible n-dimensional array."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape dimensions of the array."""
        ...

    @property
    def dtype(self) -> Any:
        """Provide the element data type of the array."""
        ...

    def tolist(self) -> Any:
        """Convert the array into nested Python lists."""
        ...

    def reshape(self, *shape: Any, **kwargs: Any) -> Any:
        """Reshape the array without changing its data."""
        ...

    def transpose(self, *axes: Any) -> Any:
        """Permute the dimensions of the array."""
        ...

    def astype(self, dtype: Any, *args: Any, **kwargs: Any) -> Any:
        """Return a copy of the array cast to a specified type."""
        ...

    def flatten(self) -> Any:
        """Return a copy of the array collapsed into one dimension."""
        ...

    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        """Return a NumPy array representation."""
        ...


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocol representing a text/SQL tokenizer."""

    @overload
    def encode(self, text: str) -> list[int]:
        """Encode a single string into a list of integer token IDs."""
        ...

    @overload
    def encode(self, text: list[str] | tuple[str, ...]) -> list[list[int]]:
        """Encode a sequence of strings into a list of integer token ID lists."""
        ...

    def encode(self, text: str | Sequence[str]) -> list[int] | list[list[int]]:
        """Encode a text string or sequence of strings into integer token IDs.

        Args:
            text: Input text string or sequence of strings to tokenize.

        Returns:
            List of integer token IDs or list of token ID lists.

        """
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode a sequence of integer token IDs back into a string.

        Args:
            token_ids: Sequence of integer token IDs.

        Returns:
            Decoded text string.

        """
        ...


@runtime_checkable
class HFTokenizerProtocol(Protocol):
    """Protocol representing a Hugging Face PreTrainedTokenizer instance."""

    def encode(self, text: str, **kwargs: Any) -> Sequence[int]:
        """Encode a text string into a sequence of integer token IDs.

        Args:
            text: Input text string to tokenize.
            **kwargs: Optional keyword arguments (such as `add_special_tokens`).

        Returns:
            Sequence of integer token IDs.

        """
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of integer token IDs back into a string.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded text string.

        """
        ...

    def get_vocab(self) -> Mapping[str, int]:
        """Return the vocabulary mapping token strings to integer IDs."""
        ...


@runtime_checkable
class TensorProtocol(Protocol):
    """Protocol defining the interface for framework tensor objects (JAX, PyTorch, TF, MLX)."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the shape of the tensor."""
        ...

    @property
    def ndim(self) -> int:
        """Provide the number of dimensions of the tensor."""
        ...

    @property
    def dtype(self) -> object:
        """Provide the data type of the tensor."""
        ...

    def reshape(self, *shape: Any, **kwargs: Any) -> Any:
        """Reshape the tensor to the given dimensions."""
        ...

    def transpose(self, *axes: Any, **kwargs: Any) -> Any:
        """Transpose the tensor axes."""
        ...

    def astype(self, dtype: Any, *args: Any, **kwargs: Any) -> Any:
        """Cast the tensor to a different data type."""
        ...

    def mean(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the arithmetic mean of the tensor elements."""
        ...

    def item(self, *args: Any) -> Any:
        """Extract a Python scalar from a single-element tensor."""
        ...

    def __getitem__(self, item: Any, /) -> Self:
        """Slice or index into the tensor."""
        ...

    def __sub__(self, other: Any, /) -> Self:
        """Subtract another tensor or scalar from this tensor."""
        ...

    def __add__(self, other: Any, /) -> Self:
        """Add another tensor or scalar to this tensor."""
        ...

    def __mul__(self, other: Any, /) -> Self:
        """Multiply this tensor by another tensor or scalar."""
        ...

    def __rmul__(self, other: Any, /) -> Self:
        """Right-multiply this tensor by a scalar or tensor."""
        ...

    def __neg__(self) -> Self:
        """Negate the tensor elementwise."""
        ...


TensorType = TensorProtocol


@runtime_checkable
class CausalLMProtocol(Protocol):
    """Protocol for causal language models supporting forward pass and autoregressive generation."""

    def __call__(self, *args: object, **kwargs: object) -> TensorProtocol | object:
        """Execute the forward pass of the causal language model."""
        ...

    def generate(self, *args: object, **kwargs: object) -> TensorProtocol | Sequence[int] | str | object:
        """Generate token sequences autoregressively."""
        ...


@runtime_checkable
class TorchModelProtocol(CausalLMProtocol, Protocol):
    """Focused protocol for PyTorch causal language models."""

    def parameters(self) -> Iterable[TensorProtocol]:
        """Yield trainable parameter tensors."""
        ...

    def named_modules(self) -> Iterable[tuple[str, Any]]:
        """Yield `(name, module)` pairs across the module hierarchy."""
        ...

    def train(self, mode: bool = True) -> Self:
        """Toggle training mode on the PyTorch model."""
        ...

    def to(self, *args: object, **kwargs: object) -> Self:
        """Move and/or cast the parameters and buffers."""
        ...

    def state_dict(self) -> dict[str, TensorProtocol]:
        """Return a dictionary containing references to the whole state of the module."""
        ...


TorchCausalLMProtocol = TorchModelProtocol


@runtime_checkable
class KerasModelProtocol(CausalLMProtocol, Protocol):
    """Focused protocol for Keras 3 / KerasNLP causal language models."""

    def compile(self, *args: object, **kwargs: object) -> None:
        """Compile the model with loss and optimizer."""
        ...

    def fit(self, *args: object, **kwargs: object) -> Any:
        """Fit the model on a dataset iterator."""
        ...

    @property
    def layers(self) -> Any:
        """Provide the sequence of Keras layers."""
        ...

    @property
    def sampler(self) -> Any:
        """Provide the model's token sampler."""
        ...

    @property
    def preprocessor(self) -> Any:
        """Provide the model's input preprocessor."""
        ...

    def save(self, *args: object, **kwargs: object) -> Any:
        """Save model weights or checkpoint to disk."""
        ...

    def load_weights(self, *args: object, **kwargs: object) -> Any:
        """Load serialized weights into the model."""
        ...


KerasCausalLMProtocol = KerasModelProtocol


@runtime_checkable
class FlaxNNXModelProtocol(Protocol):
    """Focused protocol for JAX/Flax NNX causal language models."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the forward pass of the NNX module."""
        ...

    @property
    def config(self) -> Any:
        """Provide the model configuration object."""
        ...

    @property
    def model(self) -> Any:
        """Provide the inner transformer trunk module."""
        ...

    @property
    def lm_head(self) -> Any:
        """Provide the language modeling output head."""
        ...


JaxModelProtocol = FlaxNNXModelProtocol
JaxCausalLMProtocol = FlaxNNXModelProtocol


@runtime_checkable
class MaxTextModelProtocol(Protocol):
    """Focused protocol for stateless Flax Linen / MaxText models."""

    def init(self, *args: object, **kwargs: object) -> dict[str, TensorProtocol]:
        """Initialize stateless model parameters with a PRNGKey and sample input."""
        ...

    def apply(self, *args: object, **kwargs: object) -> Any:
        """Apply a stateless transformation or forward pass."""
        ...


MaxTextCausalLMProtocol = MaxTextModelProtocol


@runtime_checkable
class MLXModelProtocol(Protocol):
    """Focused protocol for Apple Silicon MLX models."""

    def __call__(self, *args: object, **kwargs: object) -> Any:
        """Execute the forward pass of the MLX model."""
        ...

    def parameters(self) -> Any:
        """Return the parameter tree of the MLX model."""
        ...

    def update(self, parameters: dict[str, Any]) -> Self | None:
        """Update the parameters of the MLX model with a parameter tree."""
        ...

    def freeze(self) -> None:
        """Freeze base model parameters prior to adapter training."""
        ...


MlxModelProtocol = MLXModelProtocol
MlxCausalLMProtocol = MLXModelProtocol


@runtime_checkable
class EmbedderModelProtocol(Protocol):
    """Focused protocol for text and schema embedding models."""

    def encode(self, *args: object, **kwargs: object) -> Any:
        """Encode text or features into dense vector embeddings."""
        ...


@runtime_checkable
class TrainableModelProtocol(CausalLMProtocol, Protocol):
    """Protocol for trainable models across JAX, PyTorch, Keras, MaxText, and MLX."""

    def compile(self, *args: object, **kwargs: object) -> None:
        """Compile the model with loss and optimizer (Keras)."""
        ...

    def fit(self, *args: object, **kwargs: object) -> Any:
        """Fit the model on a dataset iterator (Keras)."""
        ...

    def parameters(self) -> Iterable[Any]:
        """Yield trainable parameter tensors (PyTorch/MLX)."""
        ...

    def train(self, mode: bool = True) -> Any:
        """Toggle training mode on the model (PyTorch)."""
        ...

    def apply(self, *args: object, **kwargs: object) -> Any:
        """Apply a stateless transformation or forward pass (Flax/MaxText)."""
        ...

    def load_weights(self, *args: object, **kwargs: object) -> Any:
        """Load serialized weights into the model."""
        ...

    def save(self, *args: object, **kwargs: object) -> Any:
        """Save model weights or checkpoint to disk (Keras)."""
        ...

    def freeze(self) -> None:
        """Freeze base model parameters prior to adapter training (MLX)."""
        ...


@runtime_checkable
class QuantizableModelProtocol(TrainableModelProtocol, Protocol):
    """Protocol for models supporting weight quantization and layer inspection."""

    def named_modules(self) -> Iterable[tuple[str, Any]]:
        """Yield `(name, module)` pairs across the model hierarchy."""
        ...

    @property
    def layers(self) -> Any:
        """Provide the list of model layers."""
        ...

    @property
    def sampler(self) -> Any:
        """Provide the model's token sampler."""
        ...

    @property
    def preprocessor(self) -> Any:
        """Provide the model's input preprocessor."""
        ...

    @property
    def model(self) -> Any:
        """Provide the inner trunk model."""
        ...

    @property
    def config(self) -> Any:
        """Provide the model configuration object."""
        ...

    def encode(self, *args: object, **kwargs: object) -> Any:
        """Encode text or features into embeddings."""
        ...


@runtime_checkable
class CursorProtocol(Protocol):
    """Protocol representing a DB-API 2.0 compatible database cursor."""

    def execute(self, query: str, params: Any = ...) -> Any:
        """Execute a SQL query on the cursor."""
        ...

    def fetchall(self) -> list[tuple[JSONPrimitive, ...]]:
        """Fetch all remaining rows of a query result set."""
        ...

    def close(self) -> Any:
        """Close the cursor."""
        ...


@runtime_checkable
class DatabaseConnectionProtocol(Protocol):
    """Protocol representing a DB-API 2.0 or analytical database connection."""

    def cursor(self) -> CursorProtocol:
        """Return a new `CursorProtocol` object using the connection."""
        ...

    def execute(self, query: str, params: Any = ...) -> Any:
        """Execute a SQL statement directly on the connection."""
        ...

    def commit(self) -> Any:
        """Commit any pending transaction to the database."""
        ...

    def rollback(self) -> Any:
        """Roll back to the start of any pending transaction."""
        ...

    def close(self) -> Any:
        """Close the database connection immediately."""
        ...


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
    """Internal metadata record stored in `_QUANTIZED_ARTIFACT_WEAK_REGISTRY` without a strong model cycle."""

    method: str
    is_quantized: bool
    scales: dict[str, float]


_QUANTIZED_ARTIFACT_WEAK_REGISTRY: weakref.WeakKeyDictionary[Any, _QuantizedMetadata] = weakref.WeakKeyDictionary()
_QUANTIZED_ARTIFACT_FALLBACK_REGISTRY: dict[int, QuantizedModelArtifact[Any]] = {}


def clear_quantized_artifact_registry() -> None:
    """Clear all cached entries from the quantized model artifact registry."""
    _QUANTIZED_ARTIFACT_WEAK_REGISTRY.clear()
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
    artifact = QuantizedModelArtifact(
        model=model,
        method=method,
        is_quantized=True,
        scales=resolved_scales,
    )
    try:
        _QUANTIZED_ARTIFACT_WEAK_REGISTRY[model] = _QuantizedMetadata(
            method=method,
            is_quantized=True,
            scales=resolved_scales,
        )
    except TypeError:
        _QUANTIZED_ARTIFACT_FALLBACK_REGISTRY[id(model)] = artifact
    return artifact


def get_quantized_artifact(model: TQuantModel) -> QuantizedModelArtifact[TQuantModel] | None:
    """Retrieve the registered `QuantizedModelArtifact` for a quantized model instance.

    Args:
        model: Target model instance to inspect.

    Returns:
        Registered `QuantizedModelArtifact` if the model was quantized, or `None`.

    """
    try:
        meta = _QUANTIZED_ARTIFACT_WEAK_REGISTRY.get(model)
        if meta is not None:
            return QuantizedModelArtifact(
                model=model,
                method=meta.method,
                is_quantized=meta.is_quantized,
                scales=dict(meta.scales),
            )
    except TypeError:
        pass
    return _QUANTIZED_ARTIFACT_FALLBACK_REGISTRY.get(id(model))


TrainableModelType = Union[
    JaxCausalLMProtocol,
    TorchCausalLMProtocol,
    KerasCausalLMProtocol,
    MaxTextCausalLMProtocol,
    MlxCausalLMProtocol,
    TrainableModelProtocol,
]

QuantizableModelType = Union[
    JaxCausalLMProtocol,
    TorchCausalLMProtocol,
    KerasCausalLMProtocol,
    MaxTextCausalLMProtocol,
    MlxCausalLMProtocol,
    QuantizableModelProtocol,
]

ModelType = Union[
    TrainableModelType,
    EmbedderModelProtocol,
    QuantizableModelProtocol,
]

from gemma_4_sql.backends.lazy_loader import optional_attr

ndarray: type[Any] = optional_attr("numpy", "ndarray") or object

ImageInput = Union[str, Path, bytes, Sequence[float], NDArrayProtocol, None]
AudioInput = Union[str, Path, bytes, Sequence[float], NDArrayProtocol, None]


class MultimodalInput(TypedDict, total=False):
    """Structured multimodal input containing prompt and optional media modalities."""

    prompt: str
    image: ImageInput
    audio: AudioInput
    image_token_mask: list[bool] | None
    audio_token_mask: list[bool] | None
    modality: str


class TrainResultDict(TypedDict):
    """Structured return dictionary for training, pretraining, SFT, and post-training jobs."""

    backend: str
    action: str
    model: str
    dataset: str
    epochs: int
    learning_rate: NotRequired[float]
    status: str
    final_loss: float
    distributed_strategy: NotRequired[str]
    checkpoint_dir: NotRequired[str]
    error: NotRequired[str]


class DPOResultDict(TypedDict):
    """Structured return dictionary for Direct Preference Optimization (DPO) runs."""

    backend: str
    action: str
    model: str
    dataset: str
    beta: float
    epochs: NotRequired[int]
    status: str
    final_loss: float
    error: NotRequired[str]


class InferenceResultDict(TypedDict):
    """Structured return dictionary for SQL generation and inference."""

    backend: str
    model: str
    prompt: str
    sql: str
    status: str
    beam_width: int
    confidence_score: float
    error: NotRequired[str]


class EvaluationMetricsDict(TypedDict):
    """Structured dictionary of Text-to-SQL evaluation metrics."""

    exact_match: float
    execution_accuracy: float
    valid_sql: float
    mean_confidence: NotRequired[float]


class EvaluationResultDict(TypedDict):
    """Structured return dictionary for end-to-end model evaluation."""

    backend: str
    model: str
    dataset: str
    metrics: EvaluationMetricsDict
    status: str
    modality: NotRequired[str]
    image_path: NotRequired[str]
    audio_path: NotRequired[str]


class AgentAttemptDict(TypedDict):
    """Record of a single SQL generation and execution attempt within the agentic loop."""

    attempt: int
    prompt: NotRequired[str]
    sql: str
    success: NotRequired[bool]
    error: str | None
    confidence: NotRequired[float]


class AgentLoopResultDict(TypedDict):
    """Structured return dictionary for the self-correcting SQL agent loop."""

    backend: NotRequired[str]
    model: NotRequired[str]
    initial_prompt: NotRequired[str]
    prompt: NotRequired[str]
    sql: NotRequired[str]
    final_sql: str
    results: list[tuple[JSONPrimitive, ...]]
    success: NotRequired[bool]
    attempts: int
    history: list[AgentAttemptDict]
    status: str
    error: NotRequired[str | None]
    confidence_score: NotRequired[float]


AgentResultDict = AgentLoopResultDict


class ChatMessageDict(TypedDict):
    """Single message turn in a multi-turn conversational SQL session."""

    role: str
    content: str


class ChatTurnResultDict(TypedDict):
    """Structured return dictionary for a conversational SQL chat turn."""

    backend: str
    model: str
    response: str
    history: list[ChatMessageDict]
    status: str


class QuantizeResultDict(TypedDict):
    """Structured return dictionary for model quantization pipelines."""

    backend: str
    model: str
    method: str
    status: str
    memory_reduction_factor: float
    export_path: NotRequired[str]
    quantized_layers_count: NotRequired[int]
    metadata: NotRequired[dict[str, Any]]
    error: NotRequired[str]


class PEFTResultDict(TypedDict):
    """Structured return dictionary for PEFT / LoRA injection."""

    backend: str
    action: str
    model: str
    target_modules: list[str]
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    status: str
    injected_modules: NotRequired[int]
    error: NotRequired[str]


class ExportResultDict(TypedDict):
    """Structured return dictionary for model checkpoint export."""

    backend: str
    model: str
    export_path: str
    file_path: str
    status: str
    format: str


class LatencyStatsDict(TypedDict):
    """Latency distribution statistics in milliseconds."""

    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    median_ms: NotRequired[float]
    std_ms: NotRequired[float]


class BenchmarkResultDict(TypedDict):
    """Structured return dictionary for hardware benchmarking."""

    backend: str
    model: str
    hardware: str
    batch_size: int
    tokens_per_sec: float
    latency_ms: float
    memory_mb: float
    rss_memory_mb: NotRequired[float]
    status: str
    latency_stats: NotRequired[LatencyStatsDict]


@runtime_checkable
class FastAPIAppProtocol(Protocol):
    """Protocol representing an ASGI / FastAPI application instance."""

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Handle an ASGI request scope."""
        ...

    def post(self, path: str, *args: object, **kwargs: object) -> Callable[..., Any]:
        """Register a POST route handler."""
        ...

    def get(self, path: str, *args: object, **kwargs: object) -> Callable[..., Any]:
        """Register a GET route handler."""
        ...


TBatchItem_co = TypeVar("TBatchItem_co", covariant=True)


@runtime_checkable
class DataLoaderProtocol(Iterable[TBatchItem_co], Protocol[TBatchItem_co]):
    """Protocol representing an iterable batch data loader."""


class ServeResultDict(TypedDict):
    """Structured return dictionary for continuous batching server initialization."""

    backend: str
    model: str
    port: int
    max_batch_size: int
    status: str
    mode: NotRequired[str]
    app: NotRequired[FastAPIAppProtocol | None]


class ETLResultDict(TypedDict):
    """Structured return dictionary for ETL dataloader pipelines."""

    dataset: str
    split: str
    status: str
    batch_size: int
    backend: str
    distributed: NotRequired[bool]
    loader: NotRequired[DataLoaderProtocol[Any] | Iterable[Any] | None]


DataloaderResultDict = ETLResultDict


class LogMetricsResultDict(TypedDict):
    """Structured return dictionary for TensorBoard metric logging."""

    backend: str
    logged_step: NotRequired[int]
    step: NotRequired[int]
    log_dir: str
    metrics: NotRequired[dict[str, float]]
    logged_metrics: NotRequired[dict[str, float]]
    status: str
    timestamp: NotRequired[float]
    step_duration_s: NotRequired[float]
    fallback_file: NotRequired[str]
    action: NotRequired[str]
    run_id: NotRequired[str]
    model: NotRequired[str]


class FewShotPromptResultDict(TypedDict):
    """Structured return dictionary for dynamic few-shot prompt construction."""

    backend: str
    model: str
    few_shot_prompt: str
    status: str
    num_examples: int


class FewShotResultDict(TypedDict):
    """Structured return dictionary for dynamic few-shot prompt construction and SQL generation."""

    backend: str
    model: str
    prompt: NotRequired[str]
    few_shot_prompt: str
    sql: NotRequired[str]
    confidence_score: NotRequired[float]
    status: str
    num_examples: NotRequired[int]


def _empty_extra_kwargs() -> dict[str, Any]:
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
    extra_kwargs: dict[str, Any] = field(default_factory=_empty_extra_kwargs)


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
    extra_kwargs: dict[str, Any] = field(default_factory=_empty_extra_kwargs)


TModel = TypeVar("TModel")
TOpt = TypeVar("TOpt")
TBatch = TypeVar("TBatch")
TParams = TypeVar("TParams")


@dataclass
class TrainingLoopState(Generic[TModel, TOpt, TBatch]):
    """Strongly-typed state container for standard supervised training loops."""

    dataloader: Iterable[TBatch] | None = None
    epochs: int = 1
    learning_rate: float = 1e-4
    model: TModel | None = None
    optimizer: TOpt | None = None
    train_step: Callable[..., Any] | None = None
    sharding: Any = None


@dataclass
class DPOLoopState(Generic[TModel, TOpt, TBatch]):
    """Strongly-typed state container for Direct Preference Optimization (DPO) loops."""

    dataloader: Iterable[TBatch] | None = None
    epochs: int = 1
    policy_model: TModel | None = None
    ref_model: TModel | None = None
    optimizer: TOpt | None = None
    beta: float = 0.1
    train_step: Callable[..., Any] | None = None


@dataclass
class MaxTextTrainerState(Generic[TParams, TOpt, TBatch]):
    """Typed trainer state for stateless JAX/MaxText parameter and optimizer PyTrees."""

    dataloader: Iterable[TBatch] | None = None
    epochs: int = 1
    train_step: Callable[..., tuple[TParams, TOpt, Any]] | None = None
    params: TParams | None = None
    policy_params: TParams | None = None
    ref_params: TParams | None = None
    opt_state: TOpt | None = None


@dataclass
class TorchTrainerState(Generic[TModel, TOpt, TBatch]):
    """Typed trainer state for PyTorch models, optimizers, and loss criteria."""

    dataloader: Iterable[TBatch] | None = None
    epochs: int = 1
    model: TModel | None = None
    optimizer: TOpt | None = None
    criterion: Callable[..., Any] | None = None
    device: str | object = "cpu"


@dataclass
class KerasTrainerState(Generic[TModel, TOpt, TBatch]):
    """Typed trainer state for Keras / TensorFlow compiled step execution."""

    dataloader: Iterable[TBatch] | None = None
    epochs: int = 1
    model: TModel | None = None
    optimizer: TOpt | None = None
    train_step: Callable[[TBatch], Any] | None = None


@dataclass
class TrainerState(Generic[TModel, TParams, TOpt, TBatch]):
    """Compact generic state container for training and DPO epoch runners."""

    dataloader: Iterable[TBatch] | None = None
    epochs: int = 1
    train_step: Callable[..., Any] | None = None
    params: TParams | None = None
    opt_state: TOpt | None = None
    policy_params: TParams | None = None
    ref_params: TParams | None = None
    policy_model: TModel | ModelType | None = None
    ref_model: TModel | ModelType | None = None
    optimizer: TOpt | None = None
    criterion: Callable[..., Any] | None = None
    device: str | object | None = None
    beta: float = 0.1


@runtime_checkable
class TrainSubmoduleProtocol(Protocol):
    """Protocol for backend `train` submodules."""

    def train_model(self, config: TrainingConfig, **kwargs: JSONValue) -> TrainResultDict:
        """Execute backend training pipeline."""
        ...


@runtime_checkable
class DPOSubmoduleProtocol(Protocol):
    """Protocol for backend `dpo` submodules."""

    def run_dpo(self, config: DPOConfig, **kwargs: JSONValue) -> DPOResultDict:
        """Execute backend DPO pipeline."""
        ...


@runtime_checkable
class ETLSubmoduleProtocol(Protocol):
    """Protocol for backend `etl` submodules."""

    def build_dataloader(self, config: ETLConfig, **kwargs: JSONValue) -> ETLResultDict:
        """Execute backend dataloader construction."""
        ...


@runtime_checkable
class ExportSubmoduleProtocol(Protocol):
    """Protocol for backend `export` submodules."""

    def export_model(self, model_name: str, export_path: str, **kwargs: JSONValue) -> ExportResultDict:
        """Execute backend model export."""
        ...


@runtime_checkable
class LoggingSubmoduleProtocol(Protocol):
    """Protocol for backend `logging` submodules."""

    def log_metrics(self, metrics: dict[str, float], step: int, log_dir: str = "logs", **kwargs: JSONValue) -> LogMetricsResultDict:
        """Execute backend metric logging."""
        ...


@runtime_checkable
class PEFTSubmoduleProtocol(Protocol):
    """Protocol for backend `peft` submodules."""

    def apply_lora(
        self,
        model_name: str,
        target_modules: list[str],
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        **kwargs: JSONValue,
    ) -> PEFTResultDict:
        """Execute backend LoRA adapter injection."""
        ...


@runtime_checkable
class QuantizeSubmoduleProtocol(Protocol):
    """Protocol for backend `quantize` submodules."""

    def quantize_model(self, model_name: str, method: str = "int8", **kwargs: JSONValue) -> QuantizeResultDict:
        """Execute backend model quantization."""
        ...


@runtime_checkable
class InferenceSubmoduleProtocol(Protocol):
    """Protocol for backend `inference` submodules."""

    def generate_sql(
        self,
        model_name: str,
        prompt: str,
        beam_width: int = 3,
        max_length: int = 50,
        **kwargs: JSONValue,
    ) -> InferenceResultDict:
        """Execute backend SQL generation."""
        ...


@runtime_checkable
class ServeSubmoduleProtocol(Protocol):
    """Protocol for backend `serve` submodules."""

    def serve_model(
        self,
        model_name: str,
        port: int = 8000,
        max_batch_size: int = 32,
        **kwargs: JSONValue,
    ) -> ServeResultDict:
        """Execute backend continuous batching server initialization."""
        ...


@runtime_checkable
class BenchmarkSubmoduleProtocol(Protocol):
    """Protocol for backend `benchmark` submodules."""

    def benchmark_model(
        self,
        model_name: str,
        hardware: str = "tpu-v5p",
        batch_size: int = 32,
        **kwargs: JSONValue,
    ) -> BenchmarkResultDict:
        """Execute backend hardware benchmark."""
        ...
