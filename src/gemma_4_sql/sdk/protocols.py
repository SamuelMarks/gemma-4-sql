"""Protocols for the SDK."""

from __future__ import annotations

import typing
from typing import Protocol


class BackendProtocol(Protocol):
    """Backend protocol interface."""

    def train_model(self: typing.Any, action: str, model_name: str, dataset_name: str) -> dict[str, object]:
        """Protocol method."""

    def generate_sql(self: typing.Any, model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50) -> dict[str, object]:
        """Protocol method."""

    def run_agentic_loop(self: typing.Any, model_name: str, prompt: str, db_path: str, db_type: str = "sqlite", ddl: str | None = None, **kwargs: object) -> dict[str, object]:  # type: ignore[return]
        """Protocol method."""
        _ = ddl
        _ = db_type
        _ = db_path
        _ = prompt
        _ = model_name
        kwargs.get("db_kwargs")

    def run_dpo(self: typing.Any, model_name: str, dataset_name: str, beta: float) -> dict[str, object]:
        """Protocol method."""

    def evaluate_model(self: typing.Any, model_name: str, dataset_name: str, db_path: str, db_type: str = "sqlite", ddl: str | None = None, **kwargs: object) -> dict[str, object]:  # type: ignore[return]
        """Protocol method."""
        _ = ddl
        _ = db_type
        _ = db_path
        _ = dataset_name
        _ = model_name
        kwargs.get("db_kwargs")
        kwargs.get("mock_predictions")
        kwargs.get("mock_truths")

    def build_dataloader(self: typing.Any, dataset_name: str, split: str, batch_size: int, tokenizer_name: str | None = None, duckdb_path: str | None = None, **kwargs: object) -> dict[str, object]:  # type: ignore[return]
        """Protocol method."""
        _ = duckdb_path
        _ = tokenizer_name
        _ = batch_size
        _ = split
        _ = dataset_name
        kwargs.get("duckdb_table")

    def export_model(self: typing.Any, model_name: str, output_path: str) -> dict[str, object]:
        """Protocol method."""

    def log_metrics(self: typing.Any, metrics: dict[str, float], step: int, log_dir: str = "logs") -> dict[str, object]:
        """Protocol method."""

    def apply_lora(self: typing.Any, model_name: str, target_modules: list[str], lora_r: int, lora_alpha: int, lora_dropout: float) -> dict[str, object]:
        """Protocol method."""

    def quantize_model(self: typing.Any, model_name: str, method: str = "int8", **kwargs: object) -> dict[str, object]:
        """Protocol method."""

    def chat_turn(self: typing.Any, model_name: str, history: list[dict[str, str]], new_prompt: str, **kwargs: object) -> dict[str, object]:
        """Protocol method."""

    def build_few_shot_prompt(self: typing.Any, model_name: str, prompt: str, examples: list[dict[str, str]], **kwargs: object) -> dict[str, object]:
        """Protocol method."""

    def serve_model(self: typing.Any, model_name: str, port: int = 8000, max_batch_size: int = 32, **kwargs: object) -> dict[str, object]:
        """Protocol method."""

    def benchmark_model(self: typing.Any, model_name: str, hardware: str = "tpu-v5p", batch_size: int = 32) -> dict[str, object]:
        """Protocol method."""
