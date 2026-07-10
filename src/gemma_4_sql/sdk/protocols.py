# Copyright 2024
"""Protocols for the SDK."""

from __future__ import annotations

import typing
from typing import Protocol

if typing.TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue


class TrainingProtocol(Protocol):
    """Training backend interface."""

    def train_model(self, action: str, model_name: str, dataset_name: str) -> JSONDict:
        """Protocol method."""
        ...

    def run_dpo(self, model_name: str, dataset_name: str, beta: float) -> JSONDict:
        """Protocol method."""
        ...

    def build_dataloader(self, dataset_name: str, split: str, batch_size: int, **kwargs: JSONValue) -> JSONDict:
        """Protocol method.
        Args:
            **kwargs: Overrides for ETL configuration (e.g., duckdb_path, duckdb_table).
        """
        ...

    def export_model(self, model_name: str, output_path: str) -> JSONDict:
        """Protocol method."""
        ...

    def log_metrics(self, metrics: dict[str, float], step: int, log_dir: str = "logs") -> JSONDict:
        """Protocol method."""
        ...

    def apply_lora(self, model_name: str, target_modules: list[str], **kwargs: JSONValue) -> JSONDict:
        """Protocol method.
        Args:
            **kwargs: Backend-specific LoRA parameters.
        """
        ...

    def quantize_model(self, model_name: str, method: str = "int8") -> JSONDict:
        """Protocol method."""
        ...


class InferenceProtocol(Protocol):
    """Inference backend interface."""

    def generate_sql(self, model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50) -> JSONDict:
        """Protocol method."""
        ...

    def serve_model(self, model_name: str, port: int = 8000, max_batch_size: int = 32, **kwargs: JSONValue) -> JSONDict:
        """Protocol method.
        Args:
            **kwargs: Underlying server and backend-specific configuration options.
        """
        ...

    def benchmark_model(self, model_name: str, hardware: str = "tpu-v5p", batch_size: int = 32) -> JSONDict:
        """Protocol method."""
        ...


class BackendProtocol(TrainingProtocol, InferenceProtocol, Protocol):
    """Unified Backend protocol interface."""
