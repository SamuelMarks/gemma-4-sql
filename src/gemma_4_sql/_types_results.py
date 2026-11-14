"""Structured result TypedDicts, serving handles, and ASGI/dataloader protocols for gemma-4-sql."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import NotRequired, Protocol, TypedDict, TypeVar, runtime_checkable

from gemma_4_sql._types_json import JSONPrimitive, JSONValue
from gemma_4_sql._types_tensors import AudioInput, ImageInput, NestedScalarList, TensorType


@runtime_checkable
class FastAPIAppProtocol(Protocol):
    """Protocol representing an ASGI / FastAPI application instance."""

    async def __call__(
        self,
        scope: MutableMapping[str, JSONValue],
        receive: Callable[[], Awaitable[MutableMapping[str, JSONValue]]],
        send: Callable[[MutableMapping[str, JSONValue]], Awaitable[None]],
    ) -> None:
        """Handle an ASGI request scope.

        Args:
            scope: ASGI connection scope mapping.
            receive: Async callable yielding incoming ASGI event messages.
            send: Async callable sending outgoing ASGI event messages.

        """
        ...

    def post(
        self,
        path: str,
        *args: str | bool | None,
        **kwargs: str | bool | None,
    ) -> Callable[..., JSONValue]:
        """Register a POST route handler.

        Args:
            path: URL route path.
            *args: Positional route registration options.
            **kwargs: Keyword route registration options.

        Returns:
            Decorator registering the route handler callable.

        """
        ...

    def get(
        self,
        path: str,
        *args: str | bool | None,
        **kwargs: str | bool | None,
    ) -> Callable[..., JSONValue]:
        """Register a GET route handler.

        Args:
            path: URL route path.
            *args: Positional route registration options.
            **kwargs: Keyword route registration options.

        Returns:
            Decorator registering the route handler callable.

        """
        ...


TBatchItem_co = TypeVar("TBatchItem_co", covariant=True)


@runtime_checkable
class DataLoaderProtocol(Iterable[TBatchItem_co], Protocol[TBatchItem_co]):
    """Protocol representing an iterable batch data loader."""

    def __len__(self) -> int:
        """Return the number of batches in the data loader.

        Returns:
            Total batch count in the loader.

        """
        ...


class MultimodalInput(TypedDict, total=False):
    """Structured multimodal input containing prompt and optional media modalities.

    Attributes:
        prompt: Formatted text prompt string.
        image: Raw or preprocessed image payload.
        audio: Raw or preprocessed audio payload.
        image_token_mask: Optional boolean mask over image placeholder token positions.
        audio_token_mask: Optional boolean mask over audio placeholder token positions.
        modality: Active modality label (`'text'`, `'image'`, `'audio'`, `'multimodal'`).

    """

    prompt: str
    image: ImageInput
    audio: AudioInput
    image_token_mask: list[bool] | None
    audio_token_mask: list[bool] | None
    modality: str


class BaseOperationResultDict(TypedDict):
    """Common base TypedDict for backend and SDK operation results.

    Attributes:
        backend: Backend engine identifier (`'jax'`, `'pytorch'`, `'keras'`, `'maxtext'`, `'mlx'`).
        status: Human-readable status string.
        status_code: Canonical `ExecutionStatus` code string (`'success'`, `'completed'`, `'failed'`, etc.).
        error_message: Structured error description populated when `status_code` indicates failure.

    """

    backend: str
    status: str
    status_code: NotRequired[str]
    error_message: NotRequired[str | None]


class TrainResultDict(BaseOperationResultDict):
    """Structured return dictionary for training, pretraining, SFT, and post-training jobs.

    Attributes:
        action: Training stage action (`'pretrain'`, `'sft'`, `'posttrain'`, or `'train'`).
        model: Target model identifier.
        dataset: Training dataset identifier.
        epochs: Number of epochs executed.
        learning_rate: Optimizer learning rate used.
        final_loss: Final epoch mean training loss.
        distributed_strategy: Optional distributed strategy name.
        checkpoint_dir: Optional output checkpoint directory.
        dataset_source: Optional resolved dataset source label.

    """

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
    """Structured return dictionary for Direct Preference Optimization (DPO) runs.

    Attributes:
        action: Operation action identifier (`'dpo'`).
        model: Target policy model identifier.
        dataset: Preference dataset identifier.
        beta: KL-divergence regularization strength.
        epochs: Optional number of epochs executed.
        final_loss: Final epoch mean DPO loss.
        dataset_source: Optional resolved dataset source label.

    """

    action: str
    model: str
    dataset: str
    beta: float
    epochs: NotRequired[int]
    final_loss: float
    dataset_source: NotRequired[str]


class InferenceResultDict(BaseOperationResultDict):
    """Structured return dictionary for SQL generation and inference.

    Attributes:
        model: Target model identifier.
        prompt: Input natural language prompt.
        sql: Generated SQL query string.
        beam_width: Beam search width used during decoding.
        confidence_score: Normalized decoding confidence score in `[0.0, 1.0]`.

    """

    model: str
    prompt: str
    sql: str
    beam_width: int
    confidence_score: float


class EvaluationMetricsDict(TypedDict):
    """Structured dictionary of Text-to-SQL evaluation metrics.

    Attributes:
        exact_match: Exact normalized SQL string match rate in `[0.0, 1.0]`.
        execution_accuracy: Live database result-set equivalence rate in `[0.0, 1.0]`.
        valid_sql: Syntactically executable SQL rate in `[0.0, 1.0]`.
        mean_confidence: Optional mean model confidence score across evaluated examples.

    """

    exact_match: float
    execution_accuracy: float
    valid_sql: float
    mean_confidence: NotRequired[float]


class EvaluationResultDict(TypedDict):
    """Structured return dictionary for end-to-end model evaluation.

    Attributes:
        backend: Backend engine identifier.
        model: Evaluated model identifier.
        dataset: Evaluation dataset identifier.
        metrics: Computed `EvaluationMetricsDict` values.
        status: Execution status string.
        modality: Optional input modality label.
        image_path: Optional image path used during multimodal evaluation.
        audio_path: Optional audio path used during multimodal evaluation.

    """

    backend: str
    model: str
    dataset: str
    metrics: EvaluationMetricsDict
    status: str
    modality: NotRequired[str]
    image_path: NotRequired[str]
    audio_path: NotRequired[str]


class AgentAttemptDict(TypedDict):
    """Record of a single SQL generation and execution attempt within the agentic loop.

    Attributes:
        attempt: 1-based attempt index.
        prompt: Optional prompt submitted to the model on this attempt.
        sql: Candidate SQL query generated on this attempt.
        success: Optional flag indicating whether database execution succeeded.
        error: Database or confidence-rejection error message, or `None` on success.
        confidence: Optional decoding confidence score for this attempt.

    """

    attempt: int
    prompt: NotRequired[str]
    sql: str
    success: NotRequired[bool]
    error: str | None
    confidence: NotRequired[float]


class AgentLoopResultDict(BaseOperationResultDict):
    """Structured return dictionary for the self-correcting SQL agent loop.

    Attributes:
        model: Target model identifier.
        initial_prompt: Original natural language user question.
        final_sql: Final SQL query produced by the agent loop.
        results: Database result rows returned by `final_sql` when `success` is `True`.
        success: Whether a valid executable SQL query was found within the retry budget.
        attempts: Total number of generation attempts executed.
        history: Chronological list of `AgentAttemptDict` records.
        confidence_score: Optional confidence score of the final attempt.

    """

    model: NotRequired[str]
    initial_prompt: str
    final_sql: str
    results: list[tuple[JSONPrimitive, ...]]
    success: bool
    attempts: int
    history: list[AgentAttemptDict]
    confidence_score: NotRequired[float]


AgentResultDict = AgentLoopResultDict


class ChatMessageDict(TypedDict):
    """Single message turn in a multi-turn conversational SQL session.

    Attributes:
        role: Speaker role (`'user'`, `'assistant'`, `'model'`, or `'system'`).
        content: Text or SQL content of the turn.

    """

    role: str
    content: str


class ChatTurnResultDict(TypedDict):
    """Structured return dictionary for a conversational SQL chat turn.

    Attributes:
        backend: Backend engine identifier.
        model: Target model identifier.
        response: Generated SQL response for the latest user message.
        history: Updated multi-turn conversation history including the new turn.
        status: Execution status string.

    """

    backend: str
    model: str
    response: str
    history: list[ChatMessageDict]
    status: str


class QuantizeResultDict(BaseOperationResultDict):
    """Structured return dictionary for model quantization pipelines.

    Attributes:
        model: Target model identifier.
        method: Applied quantization method (`'int8'`, `'int4'`, `'awq'`, `'gptq'`, `'gguf'`).
        memory_reduction_factor: Estimated fractional memory footprint reduction in `[0.0, 1.0)`.
        export_path: Optional output artifact path (e.g., serialized `.gguf` file).
        quantized_layers_count: Optional count of quantized linear/parameter layers.
        metadata: Optional JSON-serializable quantization metadata.

    """

    model: str
    method: str
    memory_reduction_factor: float
    export_path: NotRequired[str]
    quantized_layers_count: NotRequired[int]
    metadata: NotRequired[dict[str, JSONValue]]


class PEFTResultDict(BaseOperationResultDict):
    """Structured return dictionary for PEFT / LoRA injection.

    Attributes:
        action: Operation action identifier (`'apply_lora'`).
        model: Target model identifier.
        target_modules: Module name substrings targeted for LoRA injection.
        lora_r: Low-rank adapter dimension `r`.
        lora_alpha: LoRA scaling numerator `alpha`.
        lora_dropout: Dropout rate applied on LoRA inputs.
        injected_modules: Optional count of modules wrapped with LoRA adapters.

    """

    action: str
    model: str
    target_modules: list[str]
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    injected_modules: NotRequired[int]


class ExportResultDict(BaseOperationResultDict):
    """Structured return dictionary for model checkpoint export.

    Attributes:
        model: Exported model identifier.
        export_path: Target export directory path.
        file_path: Primary exported checkpoint/weights file path.
        format: Export serialization format (`'safetensors'`, `'orbax/saved_model'`, `'keras'`, etc.).

    """

    model: str
    export_path: str
    file_path: str
    format: str


class LatencyStatsDict(TypedDict):
    """Latency distribution statistics in milliseconds.

    Attributes:
        mean_ms: Arithmetic mean step latency in milliseconds.
        p50_ms: 50th percentile (median) latency in milliseconds.
        p90_ms: 90th percentile latency in milliseconds.
        p99_ms: 99th percentile latency in milliseconds.
        min_ms: Minimum observed latency in milliseconds.
        max_ms: Maximum observed latency in milliseconds.
        median_ms: Optional explicit median latency alias in milliseconds.
        std_ms: Optional sample standard deviation in milliseconds.

    """

    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    median_ms: NotRequired[float]
    std_ms: NotRequired[float]


class BenchmarkResultDict(BaseOperationResultDict):
    """Structured return dictionary for hardware benchmarking.

    Attributes:
        model: Benchmarked model identifier.
        hardware: Target hardware label (`'cpu'`, `'cuda'`, `'mps'`, `'tpu-v5p'`, etc.).
        batch_size: Profiled batch size.
        tokens_per_sec: Measured decoding/forward throughput in tokens per second.
        latency_ms: Mean per-batch latency in milliseconds.
        memory_mb: Peak accelerator/process memory usage in megabytes.
        rss_memory_mb: Optional host RSS memory footprint in megabytes.
        latency_stats: Optional percentile latency distribution breakdown.

    """

    model: str
    hardware: str
    batch_size: int
    tokens_per_sec: float
    latency_ms: float
    memory_mb: float
    rss_memory_mb: NotRequired[float]
    latency_stats: NotRequired[LatencyStatsDict]


class ServeResultDict(BaseOperationResultDict):
    """JSON-serializable return dictionary for continuous batching server initialization.

    Attributes:
        model: Served model identifier.
        port: Configured listening TCP port.
        max_batch_size: Maximum batch size per continuous batching dispatch window.
        mode: Serving execution mode (e.g., `'continuous_batching'`).

    """

    model: str
    port: int
    max_batch_size: int
    mode: NotRequired[str]


@dataclass(frozen=True)
class ServeHandle:
    """In-process serving handle pairing JSON-serializable `ServeResultDict` metadata with the live ASGI app.

    Attributes:
        metadata: Pure JSON-serializable `ServeResultDict` status and server configuration.
        app: Live ASGI / FastAPI application instance when initialized, or `None` on failure.

    """

    metadata: ServeResultDict
    app: FastAPIAppProtocol | None = None


type ETLBatchValue = TensorType | Sequence[int] | Sequence[Sequence[int]] | NestedScalarList


class ETLResultDict(BaseOperationResultDict):
    """Structured return dictionary for ETL dataloader pipelines.

    Attributes:
        dataset: Loaded dataset name or path.
        split: Dataset split identifier.
        batch_size: Configured batch size.
        distributed: Optional boolean flag indicating distributed sharding.
        loader: Optional constructed batch iterable or `DataLoaderProtocol`.

    """

    dataset: str
    split: str
    batch_size: int
    distributed: NotRequired[bool]
    loader: NotRequired[DataLoaderProtocol[Mapping[str, ETLBatchValue]] | Iterable[Mapping[str, ETLBatchValue]] | None]


DataloaderResultDict = ETLResultDict


class LogMetricsResultDict(BaseOperationResultDict):
    """Structured return dictionary for TensorBoard and JSONL metric logging.

    Attributes:
        step: Logged training or evaluation step index.
        log_dir: Output directory where logs were written.
        metrics: Logged metric key-value mapping.
        timestamp: Optional UNIX epoch timestamp of the log event.
        step_duration_s: Optional step duration in seconds.
        fallback_file: Optional path to `metrics.jsonl` when TensorBoard is unavailable.
        action: Optional operation action identifier (`'log_metrics'`).
        run_id: Optional experiment run identifier.
        model: Optional model identifier associated with the metrics.
        rank: Optional distributed worker rank.
        world_size: Optional distributed world size.
        status_detail: Optional human-readable backend logging detail.

    """

    step: int
    log_dir: str
    metrics: dict[str, float]
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
    """Structured return dictionary for dynamic few-shot prompt construction.

    Attributes:
        backend: Validated target backend identifier.
        model: Target model identifier.
        few_shot_prompt: Assembled prompt string containing in-context SQL examples.
        status: Operation status string.
        num_examples: Number of demonstration examples included in `few_shot_prompt`.

    """

    backend: str
    model: str
    few_shot_prompt: str
    status: str
    num_examples: int


class FewShotResultDict(TypedDict):
    """Structured return dictionary for dynamic few-shot prompt construction and SQL generation.

    Attributes:
        backend: Backend engine identifier.
        model: Target model identifier.
        prompt: Optional original user query string.
        few_shot_prompt: Assembled few-shot prompt passed to `generate_sql`.
        sql: Optional generated SQL query string.
        confidence_score: Optional decoding confidence score.
        status: Operation status string.
        num_examples: Optional count of few-shot examples included.

    """

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
    "ServeHandle",
    "ServeResultDict",
    "TBatchItem_co",
    "TrainResultDict",
]
