"""Database connection, cursor, and backend submodule protocols for gemma-4-sql."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, Self, runtime_checkable

from gemma_4_sql._types_configs import DPOConfig, ETLConfig, TrainingConfig
from gemma_4_sql._types_json import JSONPrimitive, JSONValue
from gemma_4_sql._types_results import (
    BenchmarkResultDict,
    DPOResultDict,
    ETLResultDict,
    ExportResultDict,
    InferenceResultDict,
    LogMetricsResultDict,
    PEFTResultDict,
    QuantizeResultDict,
    ServeResultDict,
    TrainResultDict,
)


@runtime_checkable
class CursorProtocol(Protocol):
    """Protocol representing a DB-API 2.0 compatible database cursor."""

    @property
    def description(self) -> Sequence[Sequence[str | int | None]] | bool | None:
        """Provide the cursor column description metadata."""
        ...

    def execute(
        self,
        query: str,
        params: Sequence[JSONPrimitive] = (),
        /,
    ) -> Self:
        """Execute a SQL query on the cursor."""
        ...

    def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
        """Fetch all remaining rows of a query result set."""
        ...

    def close(self) -> None:
        """Close the cursor."""
        ...


@runtime_checkable
class DatabaseConnectionProtocol(Protocol):
    """Protocol representing a DB-API 2.0 or analytical database connection."""

    def cursor(self) -> CursorProtocol:
        """Return a new `CursorProtocol` object using the connection."""
        ...

    def execute(
        self,
        query: str,
        params: Sequence[JSONPrimitive] = (),
        /,
    ) -> CursorProtocol:
        """Execute a SQL statement directly on the connection."""
        ...

    def commit(self) -> Self | None:
        """Commit any pending transaction to the database."""
        ...

    def rollback(self) -> Self | None:
        """Roll back any pending transaction."""
        ...

    def close(self) -> None:
        """Close the database connection immediately."""
        ...


@runtime_checkable
class AsyncDatabaseConnectionProtocol(Protocol):
    """Protocol representing an asynchronous database connection (`asyncpg`, `aiosqlite`, etc.)."""

    async def close(self) -> None:
        """Close the asynchronous database connection."""
        ...


@runtime_checkable
class DuckDBConnectionProtocol(DatabaseConnectionProtocol, Protocol):
    """Protocol for a DuckDB connection supporting `.create_function(...)` and `.cursor()`."""

    @property
    def description(self) -> Sequence[Sequence[str | int | None]] | bool | None:
        """Provide the cursor column description metadata."""
        ...

    def create_function(
        self,
        name: str,
        function: Callable[..., str],
    ) -> Self | None:
        """Register a Python scalar UDF with DuckDB."""
        ...

    def execute(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | Mapping[str, JSONPrimitive] | None = ...,
        /,
    ) -> Self:
        """Execute a SQL statement directly on the DuckDB connection."""
        ...

    def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
        """Fetch all remaining rows of a query result set."""
        ...

    def cursor(self) -> DuckDBConnectionProtocol:
        """Return a thread-local duplicate DuckDB cursor/connection."""
        ...


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


__all__ = [
    "AsyncDatabaseConnectionProtocol",
    "BenchmarkSubmoduleProtocol",
    "CursorProtocol",
    "DPOSubmoduleProtocol",
    "DatabaseConnectionProtocol",
    "DuckDBConnectionProtocol",
    "ETLSubmoduleProtocol",
    "ExportSubmoduleProtocol",
    "InferenceSubmoduleProtocol",
    "LoggingSubmoduleProtocol",
    "PEFTSubmoduleProtocol",
    "QuantizeSubmoduleProtocol",
    "ServeSubmoduleProtocol",
    "TrainSubmoduleProtocol",
]
