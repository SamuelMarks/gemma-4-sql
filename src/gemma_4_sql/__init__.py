"""Run main package for gemma-4-sql."""

from __future__ import annotations

from gemma_4_sql.exceptions import (
    BenchmarkError,
    ChatError,
    ConfigurationError,
    DatabaseExecutionError,
    DependencyMissingError,
    ETLError,
    EvaluationError,
    ExecutionStatus,
    ExportError,
    GemmaSQLError,
    InferenceError,
    PEFTError,
    QuantizationError,
    SchemaExtractionError,
    ServingError,
    SQLSafetyViolationError,
    TrainingError,
    UnsupportedQuantizationMethodError,
)
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
from gemma_4_sql.sdk.duckdb_extension import embed_in_duckdb
from gemma_4_sql.sdk.logging import log_metrics
from gemma_4_sql.sdk.rag import extract_schema_entities, retrieve_relevant_schema
from gemma_4_sql.tokenization import SQLTokenizer

__all__ = [
    "BenchmarkError",
    "ChatError",
    "ConfigurationError",
    "DatabaseExecutionError",
    "DependencyMissingError",
    "ETLError",
    "EvaluationError",
    "ExecutionStatus",
    "ExportError",
    "GemmaSQLError",
    "InferenceError",
    "LiveDatabaseEngine",
    "PEFTError",
    "QuantizationError",
    "SQLSafetyViolationError",
    "SQLTokenizer",
    "SchemaExtractionError",
    "ServingError",
    "TrainingError",
    "UnsupportedQuantizationMethodError",
    "embed_in_duckdb",
    "extract_schema_entities",
    "log_metrics",
    "retrieve_relevant_schema",
]
