"""PyTorch-specific PEFT / LoRA implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)
torch = None
peft = None
LoraConfig = None
get_peft_model = None
AutoModelForCausalLM = None
with catch_optional_imports():
    import peft
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM


def apply_lora(model_name: str, target_modules: list[str], lora_r: int, lora_alpha: int, lora_dropout: float) -> JSONDict:
    """Apply LoRA to a model using the PyTorch backend.

    Args:
        model_name: The name of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.

    Returns:
        A dictionary containing the results.
    """
    if peft is None or torch is None or AutoModelForCausalLM is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("PyTorch PEFT dependencies are missing.")
    status = "completed"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
        if hasattr(model, "print_trainable_parameters"):  # pragma: no cover
            model.print_trainable_parameters()
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Failed to apply LoRA: ")
        status = f"failed: {e!s}"
    return {"backend": "pytorch", "action": "apply_lora", "model": model_name, "target_modules": target_modules, "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout, "status": status}
