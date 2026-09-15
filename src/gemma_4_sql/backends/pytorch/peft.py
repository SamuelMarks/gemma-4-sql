"""PyTorch-specific PEFT / LoRA implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)

try:
    import peft as _peft
    import torch as _torch
    from peft import LoraConfig as _LoraConfig
    from peft import get_peft_model as _get_peft_model
    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM

    peft: Any = _peft
    torch: Any = _torch
    LoraConfig: Any = _LoraConfig
    get_peft_model: Any = _get_peft_model
    AutoModelForCausalLM: Any = _AutoModelForCausalLM
except (ImportError, AttributeError):
    peft = None
    torch = None
    LoraConfig = None
    get_peft_model = None
    AutoModelForCausalLM = None


def apply_lora(
    model_name: str,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    **kwargs: object,
) -> JSONDict:
    """Apply LoRA to a model using the PyTorch backend.

    Args:
        model_name: The name of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If PyTorch PEFT dependencies are missing.
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
        if "output_dir" in kwargs and hasattr(model, "save_pretrained"):
            model.save_pretrained(str(kwargs["output_dir"]))
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Failed to apply LoRA: ")
        status = f"failed: {e!s}"
    return {"backend": "pytorch", "action": "apply_lora", "model": model_name, "target_modules": target_modules, "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout, "status": status}
