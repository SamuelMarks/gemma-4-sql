"""
Protocols for the SDK.
"""

from __future__ import annotations

from typing import Any, Protocol


class BackendProtocol(Protocol):
    """
    Backend protocol interface.
    """

    def train_model(
        self,
        action: str,
        model_name: str,
        dataset_name: str,
    ) -> dict[str, Any]:
        """Protocol method."""

    def generate_sql(
        self, model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50
    ) -> dict[str, Any]:
        """Protocol method."""

    def run_agentic_loop(
        self,
        model_name: str,
        prompt: str,
        db_path: str,
        db_type: str = "sqlite",
        ddl: str | None = None,
        db_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Protocol method."""

    def run_dpo(
        self,
        model_name: str,
        dataset_name: str,
        beta: float,
    ) -> dict[str, Any]:
        """Protocol method."""

    def evaluate_model(
        self,
        model_name: str,
        dataset_name: str,
        db_path: str,
        db_type: str = "sqlite",
        ddl: str | None = None,
        db_kwargs: dict[str, Any] | None = None,
        mock_predictions: list[str] | None = None,
        mock_truths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Protocol method."""

    def build_dataloader(
        self,
        dataset_name: str,
        split: str,
        batch_size: int,
        tokenizer_name: str | None = None,
        duckdb_path: str | None = None,
        duckdb_table: str | None = None,
    ) -> dict[str, Any]:
        """Protocol method."""

    def export_model(self, model_name: str, output_path: str) -> dict[str, Any]:
        """Protocol method."""

    def log_metrics(
        self, metrics: dict[str, float], step: int, log_dir: str = "logs"
    ) -> dict[str, Any]:
        """Protocol method."""

    def apply_lora(
        self,
        model_name: str,
        target_modules: list[str],
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ) -> dict[str, Any]:
        """Protocol method."""

    def quantize_model(
        self, model_name: str, method: str = "int8", **kwargs: Any
    ) -> dict[str, Any]:
        """Protocol method."""

    def chat_turn(
        self,
        model_name: str,
        history: list[dict[str, str]],
        new_prompt: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Protocol method."""

    def build_few_shot_prompt(
        self,
        model_name: str,
        prompt: str,
        examples: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Protocol method."""

    def serve_model(
        self, model_name: str, port: int = 8000, max_batch_size: int = 32, **kwargs: Any
    ) -> dict[str, Any]:
        """Protocol method."""

    def benchmark_model(
        self, model_name: str, hardware: str = "tpu-v5p", batch_size: int = 32
    ) -> dict[str, Any]:
        """Protocol method."""
