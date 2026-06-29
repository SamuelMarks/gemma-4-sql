"""Keras backend approach."""

from __future__ import annotations

from .agent import run_agentic_loop
from .benchmark import benchmark_model
from .chat import chat_turn
from .dpo import run_dpo
from .etl import build_dataloader
from .evaluate import evaluate_model
from .export import export_model
from .few_shot import build_few_shot_prompt
from .inference import generate_sql
from .logging import log_metrics
from .peft import apply_lora
from .quantize import quantize_model
from .serve import serve_model
from .train import train_model

__all__ = ["apply_lora", "benchmark_model", "build_dataloader", "build_few_shot_prompt", "chat_turn", "evaluate_model", "export_model", "generate_sql", "get_trainer", "log_metrics", "quantize_model", "run_agentic_loop", "run_dpo", "serve_model", "train_model"]


def get_trainer() -> str:
    """Return the Keras trainer identifier."""
    return "keras_trainer"
