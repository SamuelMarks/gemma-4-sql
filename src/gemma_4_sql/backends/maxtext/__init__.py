"""
MaxText backend approach.
"""

from __future__ import annotations

def get_trainer() -> str:
    """Returns the MaxText trainer identifier."""
    return "maxtext_trainer"

from .benchmark import benchmark_model
from .train import train_model
from .inference import generate_sql
from .agent import run_agentic_loop
from .dpo import run_dpo
from .evaluate import evaluate_model
from .etl import build_dataloader
from .export import export_model
from .logging import log_metrics
from .peft import apply_lora
from .quantize import quantize_model
from .chat import chat_turn
from .few_shot import build_few_shot_prompt
from .serve import serve_model
