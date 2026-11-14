"""Leaf protocol definitions for CLI subparser registration and parsed command arguments."""

from __future__ import annotations

import argparse
from typing import Protocol, runtime_checkable


@runtime_checkable
class SubparserCollectionProtocol(Protocol):
    """Protocol representing an `argparse` subparser collection supporting `add_parser`."""

    def add_parser(self, name: str, *, help: str = "") -> argparse.ArgumentParser:
        """Create and register a named subcommand `ArgumentParser`.

        Args:
            name: Subcommand CLI identifier string.
            help: Summary description displayed in `--help` output.

        Returns:
            Configured `argparse.ArgumentParser` for the registered subcommand.

        """
        ...


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
    image_path: str | None
    audio_path: str | None
    modality: str


@runtime_checkable
class TrainArgsProtocol(Protocol):
    """Protocol for parsed CLI arguments for training subcommands.

    Covers `train_cmd`, `pretrain_cmd`, `sft_cmd`, `posttrain_cmd`, and `dpo_cmd`.

    """

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
    "SubparserCollectionProtocol",
    "TrainArgsProtocol",
]
