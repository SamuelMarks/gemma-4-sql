"""Keras backend approach."""

from __future__ import annotations

from .benchmark import benchmark_model
from .dpo import run_dpo
from .etl import build_dataloader
from .export import export_model
from .inference import generate_sql
from .logging import log_metrics
from .peft import apply_lora
from .quantize import quantize_model
from .serve import serve_model
from .train import train_model

__all__ = ["apply_lora", "benchmark_model", "build_dataloader", "export_model", "generate_sql", "get_trainer", "log_metrics", "quantize_model", "run_dpo", "serve_model", "train_model"]


def get_trainer() -> str:
    """Return the Keras trainer identifier.

    Returns:
        The resulting string.
    """
    return "keras_trainer"
