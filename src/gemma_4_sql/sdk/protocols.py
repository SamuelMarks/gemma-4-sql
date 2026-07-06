# Copyright 2024
"""Protocols for the SDK."""

from __future__ import annotations  # pragma: no cover

import typing  # pragma: no cover
from typing import Protocol  # pragma: no cover

if typing.TYPE_CHECKING:  # pragma: no cover
    from gemma_4_sql.type_hints import JSONDict, JSONValue  # pragma: no cover


class TrainingProtocol(Protocol):  # pragma: no cover
    """Training backend interface."""  # pragma: no cover

    def train_model(self, action: str, model_name: str, dataset_name: str) -> JSONDict:  # pragma: no cover
        """Protocol method."""  # pragma: no cover
        ...  # pragma: no cover

    def run_dpo(self, model_name: str, dataset_name: str, beta: float) -> JSONDict:  # pragma: no cover
        """Protocol method."""  # pragma: no cover
        ...  # pragma: no cover

    def build_dataloader(self, dataset_name: str, split: str, batch_size: int, **kwargs: JSONValue) -> JSONDict:  # pragma: no cover
        """Protocol method.
        Args:
            **kwargs: Overrides for ETL configuration (e.g., duckdb_path, duckdb_table).
        """
        ...  # pragma: no cover

    def export_model(self, model_name: str, output_path: str) -> JSONDict:  # pragma: no cover
        """Protocol method."""  # pragma: no cover
        ...  # pragma: no cover

    def log_metrics(self, metrics: dict[str, float], step: int, log_dir: str = "logs") -> JSONDict:  # pragma: no cover
        """Protocol method."""  # pragma: no cover
        ...  # pragma: no cover

    def apply_lora(self, model_name: str, target_modules: list[str], **kwargs: JSONValue) -> JSONDict:  # pragma: no cover
        """Protocol method.
        Args:
            **kwargs: Backend-specific LoRA parameters.
        """
        ...  # pragma: no cover

    def quantize_model(self, model_name: str, method: str = "int8") -> JSONDict:  # pragma: no cover
        """Protocol method."""  # pragma: no cover
        ...  # pragma: no cover


class InferenceProtocol(Protocol):  # pragma: no cover
    """Inference backend interface."""  # pragma: no cover

    def generate_sql(self, model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50) -> JSONDict:  # pragma: no cover
        """Protocol method."""  # pragma: no cover
        ...  # pragma: no cover

    def serve_model(self, model_name: str, port: int = 8000, max_batch_size: int = 32, **kwargs: JSONValue) -> JSONDict:  # pragma: no cover
        """Protocol method.
        Args:
            **kwargs: Underlying server and backend-specific configuration options.
        """
        ...  # pragma: no cover

    def benchmark_model(self, model_name: str, hardware: str = "tpu-v5p", batch_size: int = 32) -> JSONDict:  # pragma: no cover
        """Protocol method."""  # pragma: no cover
        ...  # pragma: no cover


class BackendProtocol(TrainingProtocol, InferenceProtocol, Protocol):  # pragma: no cover
    """Unified Backend protocol interface."""
