"""MaxText backend approach."""

from __future__ import annotations

from .benchmark import benchmark_model
from .config_generator import (
    MaxTextHyperparameters,
    build_maxtext_cli_args,
    build_maxtext_hyperparameters,
    generate_maxtext_gin_config,
    normalize_model_architecture,
    save_maxtext_gin_config,
)
from .dpo import run_dpo
from .etl import build_dataloader
from .export import export_model
from .inference import generate_sql
from .logging import log_metrics
from .peft import apply_lora
from .quantize import quantize_model
from .serve import serve_model
from .train import save_maxtext_checkpoint, train_model

__all__ = [
    "MaxTextHyperparameters",
    "apply_lora",
    "benchmark_model",
    "build_dataloader",
    "build_maxtext_cli_args",
    "build_maxtext_hyperparameters",
    "export_model",
    "generate_maxtext_gin_config",
    "generate_sql",
    "get_trainer",
    "log_metrics",
    "normalize_model_architecture",
    "quantize_model",
    "run_dpo",
    "save_maxtext_checkpoint",
    "save_maxtext_gin_config",
    "serve_model",
    "train_model",
]


def get_trainer() -> str:
    """Return the MaxText trainer identifier.

    Returns:
        The resulting string.
    """
    return "maxtext_trainer"
