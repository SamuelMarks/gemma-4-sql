"""Modular CLI argument parser package for gemma-4-sql."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gemma_4_sql.cli_parsers.benchmark_parser import (
    add_benchmark_parser,
    add_misc_parsers,
    add_rag_log_parsers,
    add_tokenize_parser,
)
from gemma_4_sql.cli_parsers.db_parser import add_db_parsers
from gemma_4_sql.cli_parsers.etl_parser import add_etl_parsers, add_etl_subparser
from gemma_4_sql.cli_parsers.serve_parser import (
    add_evaluate_parsers,
    add_generate_agent_parsers,
    add_inference_parsers,
    add_serve_export_parsers,
)
from gemma_4_sql.cli_parsers.train_parser import (
    add_peft_quantize_parsers,
    add_training_parsers,
    add_training_subparser,
)


@runtime_checkable
class ETLArgsProtocol(Protocol):
    """Protocol for parsed CLI arguments passed to `etl_*_cmd`."""

    dataset: str
    split: str
    batch_size: int
    backend: str
    distributed: bool
    tokenizer: str | None
    duckdb_path: str | None
    duckdb_table: str | None
    image_column: str | None
    audio_column: str | None
    modality: str


@runtime_checkable
class TrainArgsProtocol(Protocol):
    """Protocol for parsed CLI arguments passed to `train_cmd`, `pretrain_cmd`, `sft_cmd`, `posttrain_cmd`, and `dpo_cmd`."""

    model: str
    dataset: str
    epochs: int
    learning_rate: float
    batch_size: int
    backend: str
    distributed_strategy: str
    image_path: str | None
    audio_path: str | None
    modality: str


@runtime_checkable
class ServeArgsProtocol(Protocol):
    """Protocol for parsed CLI arguments passed to `serve_cmd`."""

    model: str
    port: int
    max_batch_size: int
    run_server: bool
    backend: str
    image_path: str | None
    audio_path: str | None
    modality: str


@runtime_checkable
class EvaluateArgsProtocol(Protocol):
    """Protocol for parsed CLI arguments passed to `evaluate_cmd`."""

    model: str
    dataset: str
    backend: str
    db_path: str
    db_type: str
    db_kwargs: str
    ddl: str
    image_path: str | None
    audio_path: str | None
    modality: str
    max_batches: int


@runtime_checkable
class AgentArgsProtocol(Protocol):
    """Protocol for parsed CLI arguments passed to `agent_cmd`."""

    model: str
    prompt: str
    db_path: str
    db_type: str
    db_kwargs: str
    ddl: str
    max_retries: int
    min_confidence: float
    backend: str
    image_path: str | None
    audio_path: str | None
    modality: str


@runtime_checkable
class ChatArgsProtocol(Protocol):
    """Protocol for parsed CLI arguments passed to `chat_cmd`."""

    model: str
    prompt: str
    history: str
    backend: str
    image_path: str | None
    audio_path: str | None
    modality: str


@runtime_checkable
class FewShotArgsProtocol(Protocol):
    """Protocol for parsed CLI arguments passed to `few_shot_cmd`."""

    model: str
    prompt: str
    examples: str
    backend: str
    image_path: str | None
    audio_path: str | None
    modality: str


@runtime_checkable
class RagArgsProtocol(Protocol):
    """Protocol for parsed CLI arguments passed to `rag_cmd`."""

    action: str
    prompt: str
    ddl: str
    image_path: str | None
    audio_path: str | None
    modality: str


__all__ = [
    "AgentArgsProtocol",
    "ChatArgsProtocol",
    "ETLArgsProtocol",
    "EvaluateArgsProtocol",
    "FewShotArgsProtocol",
    "RagArgsProtocol",
    "ServeArgsProtocol",
    "TrainArgsProtocol",
    "add_benchmark_parser",
    "add_db_parsers",
    "add_etl_parsers",
    "add_etl_subparser",
    "add_evaluate_parsers",
    "add_generate_agent_parsers",
    "add_inference_parsers",
    "add_misc_parsers",
    "add_peft_quantize_parsers",
    "add_rag_log_parsers",
    "add_serve_export_parsers",
    "add_tokenize_parser",
    "add_training_parsers",
    "add_training_subparser",
]
