"""Database connection, cursor, and backend submodule protocols for gemma-4-sql."""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Mapping, Sequence
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
        """Provide the cursor column description metadata.

        Returns:
            DB-API 2.0 column description sequence or `None` for non-row statements.

        """
        ...

    def execute(
        self,
        query: str,
        params: Sequence[JSONPrimitive] = (),
        /,
    ) -> Self:
        """Execute a SQL query on the cursor.

        Args:
            query: SQL statement text.
            params: Positional query bind parameters.

        Returns:
            Cursor instance after execution.

        """
        ...

    def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
        """Fetch all remaining rows of a query result set.

        Returns:
            Sequence of result rows.

        """
        ...

    def close(self) -> None:
        """Close the cursor and release driver resources."""
        ...


@runtime_checkable
class DatabaseConnectionProtocol(Protocol):
    """Protocol representing a DB-API 2.0 or analytical database connection."""

    def cursor(self) -> CursorProtocol:
        """Return a new `CursorProtocol` object using the connection.

        Returns:
            Newly created database cursor.

        """
        ...

    def execute(
        self,
        query: str,
        params: Sequence[JSONPrimitive] = (),
        /,
    ) -> CursorProtocol:
        """Execute a SQL statement directly on the connection.

        Args:
            query: SQL statement text.
            params: Positional query bind parameters.

        Returns:
            Cursor holding the statement execution results.

        """
        ...

    def commit(self) -> Self | None:
        """Commit any pending transaction to the database.

        Returns:
            Optional self reference or `None`.

        """
        ...

    def rollback(self) -> Self | None:
        """Roll back any pending transaction.

        Returns:
            Optional self reference or `None`.

        """
        ...

    def close(self) -> None:
        """Close the database connection immediately."""
        ...


@runtime_checkable
class AsyncDatabaseConnectionProtocol(Protocol):
    """Protocol representing an asynchronous database connection (`asyncpg`, `aiosqlite`, etc.)."""

    def close(self) -> Coroutine[None, None, None] | None:
        """Close the asynchronous database connection.

        Returns:
            Optional awaitable coroutine when closing asynchronously, or `None`.

        """
        ...


@runtime_checkable
class DuckDBConnectionProtocol(DatabaseConnectionProtocol, Protocol):
    """Protocol for a DuckDB connection supporting `.create_function(...)` and `.cursor()`."""

    @property
    def description(self) -> Sequence[Sequence[str | int | None]] | bool | None:
        """Provide the cursor column description metadata.

        Returns:
            DuckDB result column description metadata or `None`.

        """
        ...

    def create_function(
        self,
        name: str,
        function: Callable[..., str],
    ) -> Self | None:
        """Register a Python scalar UDF with DuckDB.

        Args:
            name: SQL function identifier to register.
            function: Python callable implementing the scalar UDF.

        Returns:
            Optional DuckDB connection reference or `None`.

        """
        ...

    def execute(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | Mapping[str, JSONPrimitive] | None = ...,
        /,
    ) -> Self:
        """Execute a SQL statement directly on the DuckDB connection.

        Args:
            query: SQL query string to execute.
            params: Optional positional or named query parameters.

        Returns:
            Connection instance positioned on the result set.

        """
        ...

    def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
        """Fetch all remaining rows of a query result set.

        Returns:
            Sequence of result rows.

        """
        ...

    def cursor(self) -> DuckDBConnectionProtocol:
        """Return a thread-local duplicate DuckDB cursor/connection.

        Returns:
            Independent DuckDB cursor connection sharing the database catalog.

        """
        ...


@runtime_checkable
class TrainSubmoduleProtocol(Protocol):
    """Protocol for backend `train` submodules."""

    def train_model(self, config: TrainingConfig, **kwargs: JSONValue) -> TrainResultDict:
        """Execute backend training pipeline.

        Args:
            config: Validated `TrainingConfig` hyperparameters.
            **kwargs: Additional backend-specific options.

        Returns:
            Structured `TrainResultDict` summary.

        """
        ...


@runtime_checkable
class DPOSubmoduleProtocol(Protocol):
    """Protocol for backend `dpo` submodules."""

    def run_dpo(self, config: DPOConfig, **kwargs: JSONValue) -> DPOResultDict:
        """Execute backend DPO pipeline.

        Args:
            config: Validated `DPOConfig` hyperparameters.
            **kwargs: Additional backend-specific options.

        Returns:
            Structured `DPOResultDict` summary.

        """
        ...


@runtime_checkable
class ETLSubmoduleProtocol(Protocol):
    """Protocol for backend `etl` submodules."""

    def build_dataloader(self, config: ETLConfig, **kwargs: JSONValue) -> ETLResultDict:
        """Execute backend dataloader construction.

        Args:
            config: Validated `ETLConfig` parameters.
            **kwargs: Additional backend-specific options.

        Returns:
            Structured `ETLResultDict` containing the constructed loader.

        """
        ...


@runtime_checkable
class ExportSubmoduleProtocol(Protocol):
    """Protocol for backend `export` submodules."""

    def export_model(self, model_name: str, export_path: str, **kwargs: JSONValue) -> ExportResultDict:
        """Execute backend model export.

        Args:
            model_name: Target model name or checkpoint path.
            export_path: Destination export directory or file path.
            **kwargs: Additional backend-specific options.

        Returns:
            Structured `ExportResultDict` summary.

        """
        ...


@runtime_checkable
class LoggingSubmoduleProtocol(Protocol):
    """Protocol for backend `logging` submodules."""

    def log_metrics(
        self, metrics: dict[str, float], step: int, log_dir: str = "logs", **kwargs: JSONValue
    ) -> LogMetricsResultDict:
        """Execute backend metric logging.

        Args:
            metrics: Mapping of metric names to scalar float values.
            step: Global training or evaluation step index.
            log_dir: Output log directory path.
            **kwargs: Additional metadata fields.

        Returns:
            Structured `LogMetricsResultDict` summary.

        """
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
        """Execute backend LoRA adapter injection.

        Args:
            model_name: Target model identifier.
            target_modules: Module name substrings to adapt with LoRA.
            lora_r: Low-rank bottleneck dimension.
            lora_alpha: Scaling factor numerator.
            lora_dropout: Dropout probability on LoRA inputs.
            **kwargs: Additional backend-specific options.

        Returns:
            Structured `PEFTResultDict` summary.

        """
        ...


@runtime_checkable
class QuantizeSubmoduleProtocol(Protocol):
    """Protocol for backend `quantize` submodules."""

    def quantize_model(self, model_name: str, method: str = "int8", **kwargs: JSONValue) -> QuantizeResultDict:
        """Execute backend model quantization.

        Args:
            model_name: Target model identifier.
            method: Quantization algorithm (`'int8'`, `'int4'`, `'awq'`, `'gptq'`, `'gguf'`).
            **kwargs: Additional quantization options.

        Returns:
            Structured `QuantizeResultDict` summary.

        """
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
        """Execute backend SQL generation.

        Args:
            model_name: Target model identifier.
            prompt: Natural language SQL prompt.
            beam_width: Number of beams for search decoding.
            max_length: Maximum number of tokens to generate.
            **kwargs: Additional decoding or multimodal options.

        Returns:
            Structured `InferenceResultDict` containing generated SQL and confidence.

        """
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
        """Execute backend continuous batching server initialization.

        Args:
            model_name: Target model identifier.
            port: TCP port to listen on.
            max_batch_size: Maximum batch size per continuous batching window.
            **kwargs: Additional serving options.

        Returns:
            JSON-serializable `ServeResultDict` metadata.

        """
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
        """Execute backend hardware benchmark.

        Args:
            model_name: Target model identifier.
            hardware: Hardware target descriptor.
            batch_size: Batch size to profile.
            **kwargs: Additional benchmark parameters.

        Returns:
            Structured `BenchmarkResultDict` latency and throughput metrics.

        """
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
