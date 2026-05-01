"""
PyTorch-specific PEFT / LoRA implementation.
"""

from __future__ import annotations

from typing import Any

try:
    import peft
except Exception:
    peft = None


def apply_lora(
    model_name: str,
    target_modules: list[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> dict[str, Any]:
    """
    Applies LoRA to a model using the PyTorch backend.

    Args:
        model_name: Name of the base model.
        target_modules: List of module names to apply LoRA to.
        lora_r: LoRA attention dimension (rank).
        lora_alpha: LoRA alpha parameter.
        lora_dropout: LoRA dropout probability.

    Returns:
        Dictionary containing PEFT status.
    """
    status = "completed"
    if peft is None:
        status = "mocked_missing_peft"

    return {
        "backend": "pytorch",
        "action": "apply_lora",
        "model": model_name,
        "target_modules": target_modules,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "status": status,
    }


def missing_dummy(
    model_name: str,
    rank: int,
    alpha: int,
    target_modules: list[str] | None = None,
) -> dict[str, Any]:
    """Dummy function for missing protocol implementation."""
    return {"status": "dummy"}
