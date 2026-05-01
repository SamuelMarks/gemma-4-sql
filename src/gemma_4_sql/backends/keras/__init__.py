"""
Keras backend approach.
"""

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
from .peft import apply_lora, missing_dummy
from .quantize import quantize_model
from .serve import serve_model
from .train import train_model

__all__ = [
    "run_agentic_loop",
    "benchmark_model",
    "chat_turn",
    "run_dpo",
    "build_dataloader",
    "evaluate_model",
    "export_model",
    "build_few_shot_prompt",
    "generate_sql",
    "log_metrics",
    "apply_lora",
    "missing_dummy",
    "quantize_model",
    "serve_model",
    "train_model",
    "get_trainer",
]


def get_trainer() -> str:
    """Returns the Keras trainer identifier."""
    return "keras_trainer"
