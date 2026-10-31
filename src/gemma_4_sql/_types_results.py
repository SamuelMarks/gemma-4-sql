"""Structured result TypedDicts and ASGI/dataloader protocols for gemma-4-sql."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, MutableMapping, Sequence
from typing import NotRequired, Protocol, TypedDict, TypeVar, runtime_checkable

from gemma_4_sql._types_json import JSONPrimitive, JSONValue
from gemma_4_sql._types_tensors import AudioInput, ImageInput, TensorType


@runtime_checkable
class FastAPIAppProtocol(Protocol):
    """Protocol representing an ASGI / FastAPI application instance."""

    async def __call__(
        self,
        scope: MutableMapping[str, JSONValue],
        receive: Callable[[], Awaitable[MutableMapping[str, JSONValue]]],
        send: Callable[[MutableMapping[str, JSONValue]], Awaitable[None]],
    ) -> None:
        """Handle an ASGI request scope."""
        ...

    def post(
        self,
        path: str,
        *args: str | bool | None,
        **kwargs: str | bool | None,
    ) -> Callable[..., JSONValue]:
        """Register a POST route handler."""
        ...

    def get(
        self,
        path: str,
        *args: str | bool | None,
        **kwargs: str | bool | None,
    ) -> Callable[..., JSONValue]:
        """Register a GET route handler."""
        ...


TBatchItem_co = TypeVar("TBatchItem_co", covariant=True)


@runtime_checkable
class DataLoaderProtocol(Iterable[TBatchItem_co], Protocol[TBatchItem_co]):
    """Protocol representing an iterable batch data loader."""

    def __len__(self) -> int:
        """Return the number of batches in the data loader."""
        ...


class MultimodalInput(TypedDict, total=False):
    """Structured multimodal input containing prompt and optional media modalities."""

    prompt: str
    image: ImageInput
    audio: AudioInput
    image_token_mask: list[bool] | None
    audio_token_mask: list[bool] | None
    modality: str


class BaseOperationResultDict(TypedDict):
    """Common base TypedDict for backend and SDK operation results."""

    backend: str
    status: str
    status_code: NotRequired[str]
    error: NotRequired[str]
    error_message: NotRequired[str | None]


class TrainResultDict(BaseOperationResultDict):
    """Structured return dictionary for training, pretraining, SFT, and post-training jobs."""

    action: str
    model: str
    dataset: str
    epochs: int
    learning_rate: NotRequired[float]
    final_loss: float
    distributed_strategy: NotRequired[str]
    checkpoint_dir: NotRequired[str]
    dataset_source: NotRequired[str]


class DPOResultDict(BaseOperationResultDict):
    """Structured return dictionary for Direct Preference Optimization (DPO) runs."""

    action: str
    model: str
    dataset: str
    beta: float
    epochs: NotRequired[int]
    final_loss: float
    dataset_source: NotRequired[str]


class InferenceResultDict(BaseOperationResultDict):
    """Structured return dictionary for SQL generation and inference."""

    model: str
    prompt: str
    sql: str
    beam_width: int
    confidence_score: float


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


class QuantizeResultDict(BaseOperationResultDict):
    """Structured return dictionary for model quantization pipelines."""

    model: str
    method: str
    memory_reduction_factor: float
    export_path: NotRequired[str]
    quantized_layers_count: NotRequired[int]
    metadata: NotRequired[dict[str, JSONValue]]


class PEFTResultDict(BaseOperationResultDict):
    """Structured return dictionary for PEFT / LoRA injection."""

    action: str
    model: str
    target_modules: list[str]
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    injected_modules: NotRequired[int]


class ExportResultDict(BaseOperationResultDict):
    """Structured return dictionary for model checkpoint export."""

    model: str
    export_path: str
    file_path: str
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


class BenchmarkResultDict(BaseOperationResultDict):
    """Structured return dictionary for hardware benchmarking."""

    model: str
    hardware: str
    batch_size: int
    tokens_per_sec: float
    latency_ms: float
    memory_mb: float
    rss_memory_mb: NotRequired[float]
    latency_stats: NotRequired[LatencyStatsDict]


class ServeResultDict(BaseOperationResultDict):
    """Structured return dictionary for continuous batching server initialization."""

    model: str
    port: int
    max_batch_size: int
    mode: NotRequired[str]
    app: NotRequired[FastAPIAppProtocol | None]


class ETLResultDict(BaseOperationResultDict):
    """Structured return dictionary for ETL dataloader pipelines."""

    dataset: str
    split: str
    batch_size: int
    distributed: NotRequired[bool]
    loader: NotRequired[DataLoaderProtocol[Mapping[str, TensorType | Sequence[int]]] | Iterable[Mapping[str, TensorType | Sequence[int]]] | None]


DataloaderResultDict = ETLResultDict


class LogMetricsResultDict(BaseOperationResultDict):
    """Structured return dictionary for TensorBoard metric logging."""

    logged_step: NotRequired[int]
    step: NotRequired[int]
    log_dir: str
    metrics: NotRequired[dict[str, float]]
    logged_metrics: NotRequired[dict[str, float]]
    timestamp: NotRequired[float]
    step_duration_s: NotRequired[float]
    fallback_file: NotRequired[str]
    action: NotRequired[str]
    run_id: NotRequired[str]
    model: NotRequired[str]
    rank: NotRequired[int]
    world_size: NotRequired[int]
    status_detail: NotRequired[str]


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


__all__ = [
    "AgentAttemptDict",
    "AgentLoopResultDict",
    "AgentResultDict",
    "BaseOperationResultDict",
    "BenchmarkResultDict",
    "ChatMessageDict",
    "ChatTurnResultDict",
    "DPOResultDict",
    "DataLoaderProtocol",
    "DataloaderResultDict",
    "ETLResultDict",
    "EvaluationMetricsDict",
    "EvaluationResultDict",
    "ExportResultDict",
    "FastAPIAppProtocol",
    "FewShotPromptResultDict",
    "FewShotResultDict",
    "InferenceResultDict",
    "LatencyStatsDict",
    "LogMetricsResultDict",
    "MultimodalInput",
    "PEFTResultDict",
    "QuantizeResultDict",
    "ServeResultDict",
    "TBatchItem_co",
    "TrainResultDict",
]
