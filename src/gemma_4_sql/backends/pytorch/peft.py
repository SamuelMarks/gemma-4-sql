"""PyTorch-specific PEFT / LoRA implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict

logger = logging.getLogger(__name__)

try:
    import peft
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    torch = None
    peft = None
    LoraConfig = None
    get_peft_model = None
    AutoModelForCausalLM = None


def apply_lora(model_name: str, target_modules: list[str], lora_r: int, lora_alpha: int, lora_dropout: float) -> JSONDict:
    """Apply LoRA to a model using the PyTorch backend.

    Args:
    ----
        model_name: Name of the base model.
        target_modules: List of module names to apply LoRA to.
        lora_r: LoRA attention dimension (rank).
        lora_alpha: LoRA alpha parameter.
        lora_dropout: LoRA dropout probability.

    Returns:
    -------
        Dictionary containing PEFT status.

    """
    status = "completed"
    if peft is not None and torch is not None and AutoModelForCausalLM is not None:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, lora_config)

            # Simulated check
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()

        except Exception as e:
            logger.exception("Failed to apply LoRA: %s", e)
            status = f"failed: {e!s}"
    else:
        status = "mocked_missing_peft"

    return {"backend": "pytorch", "action": "apply_lora", "model": model_name, "target_modules": target_modules, "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout, "status": status}
