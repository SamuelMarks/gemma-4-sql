# Copyright 2024
"""Protocols for the SDK."""

from __future__ import annotations

import typing
from typing import Protocol

if typing.TYPE_CHECKING:
    from gemma_4_sql.type_hints import DPOConfig, ETLConfig, JSONDict, JSONValue, TrainingConfig


class TrainingProtocol(Protocol):
    """Training backend interface."""

    def train_model(self, config: TrainingConfig, **kwargs: JSONValue) -> JSONDict:
        """Train a model according to training configuration.

        Args:
            config: TrainingConfig specifying dataset, epochs, learning rate, and strategy.
            **kwargs: Backend-specific execution overrides.

        Returns:
            Dictionary containing training status and final metrics.
        """
        ...

    def run_dpo(self, config: DPOConfig, **kwargs: JSONValue) -> JSONDict:
        """Run Direct Preference Optimization (DPO).

        Args:
            config: DPOConfig specifying dataset, model, beta, epochs, and learning rate.
            **kwargs: Backend-specific execution overrides.

        Returns:
            Dictionary containing DPO training status and final metrics.
        """
        ...

    def build_dataloader(self, config: ETLConfig, **kwargs: JSONValue) -> JSONDict:
        """Construct ETL dataloader pipeline.

        Args:
            config: ETLConfig specifying dataset name, split, batch size, and storage options.
            **kwargs: Overrides for ETL configuration (e.g., duckdb_path, duckdb_table).

        Returns:
            Dictionary containing dataloader objects and metadata.
        """
        ...

    def export_model(self, model_name: str, export_path: str, **kwargs: JSONValue) -> JSONDict:
        """Export model weights and architecture definition.

        Args:
            model_name: Target model identifier.
            export_path: Destination path for exported artifacts.
            **kwargs: Backend-specific export options.

        Returns:
            Dictionary containing export status and target path.
        """
        ...

    def log_metrics(self, metrics: dict[str, float], step: int, log_dir: str = "logs", **kwargs: JSONValue) -> JSONDict:
        """Log training and evaluation metrics.

        Args:
            metrics: Dictionary of metric names and numeric values.
            step: Training or evaluation step index.
            log_dir: Directory path for event logs.
            **kwargs: Backend-specific logger options.

        Returns:
            Dictionary containing logging confirmation and status.
        """
        ...

    def apply_lora(
        self,
        model_name: str,
        target_modules: list[str],
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        **kwargs: JSONValue,
    ) -> JSONDict:
        """Apply Parameter-Efficient Fine-Tuning (LoRA) adapters.

        Args:
            model_name: Target model identifier.
            target_modules: Layer names to apply LoRA rank decomposition on.
            lora_r: Rank dimension of LoRA adapters.
            lora_alpha: Scaling alpha parameter.
            lora_dropout: Dropout probability for LoRA layers.
            **kwargs: Backend-specific LoRA parameters.

        Returns:
            Dictionary containing adapter configuration status.
        """
        ...

    def quantize_model(self, model_name: str, method: str = "int8", **kwargs: JSONValue) -> JSONDict:
        """Quantize model weights for reduced memory footprint.

        Args:
            model_name: Target model identifier.
            method: Quantization method (e.g., 'int8', 'awq', 'gptq').
            **kwargs: Backend-specific quantization options.

        Returns:
            Dictionary containing quantization status and reduction factor.
        """
        ...


class InferenceProtocol(Protocol):
    """Inference backend interface."""

    def generate_sql(
        self,
        model_name: str,
        prompt: str,
        beam_width: int = 3,
        max_length: int = 50,
        **kwargs: JSONValue,
    ) -> JSONDict:
        """Generate a SQL query from natural language input.

        Args:
            model_name: Target model identifier.
            prompt: Input text prompt or schema context.
            beam_width: Number of beams in beam search.
            max_length: Maximum generation token length.
            **kwargs: Backend-specific generation options.

        Returns:
            Dictionary containing generated SQL and confidence metrics.
        """
        ...

    def serve_model(
        self,
        model_name: str,
        port: int = 8000,
        max_batch_size: int = 32,
        **kwargs: JSONValue,
    ) -> JSONDict:
        """Serve model via continuous batching API endpoint.

        Args:
            model_name: Target model identifier.
            port: Network port to host the server on.
            max_batch_size: Maximum continuous batch size.
            **kwargs: Underlying server and backend-specific configuration options.

        Returns:
            Dictionary containing server status and instance details.
        """
        ...

    def benchmark_model(
        self,
        model_name: str,
        hardware: str = "tpu-v5p",
        batch_size: int = 32,
        **kwargs: typing.Any,
    ) -> JSONDict:
        """Run latency, throughput, and memory benchmarking.

        Args:
            model_name: Target model identifier.
            hardware: Target hardware device ('gpu', 'tpu-v5p', 'cpu').
            batch_size: Batch size for benchmarking.
            **kwargs: Additional benchmark options like warmup_steps or num_runs.

        Returns:
            Dictionary containing throughput (tokens/sec), latency (ms), and memory (MB).
        """
        ...


class BackendProtocol(TrainingProtocol, InferenceProtocol, Protocol):
    """Unified Backend protocol interface combining training and inference capabilities."""
