"""PyTorch-specific PEFT / LoRA implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import PEFTResultDict
logger = logging.getLogger(__name__)

peft = optional_import("peft")
torch = optional_import("torch")
LoraConfig = optional_attr("peft", "LoraConfig")
get_peft_model = optional_attr("peft", "get_peft_model")
AutoModelForCausalLM = optional_attr("transformers", "AutoModelForCausalLM")


def apply_lora(
    model_name: str,
    target_modules: list[str],
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    **kwargs: object,
) -> PEFTResultDict:
    """Apply LoRA to a model using the PyTorch backend.

    Args:
        model_name: The name of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.
        **kwargs: Additional arbitrary configuration options (including `raise_on_error`).

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If PyTorch PEFT dependencies are missing.
        TrainingError: If LoRA application fails and `raise_on_error` is enabled.

    """
    if peft is None or torch is None or AutoModelForCausalLM is None:
        raise DependencyMissingError("PyTorch PEFT dependencies are missing.", backend="pytorch")
    status = "completed"
    injected_count = 0
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
        injected_count = len(target_modules)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        if "output_dir" in kwargs and hasattr(model, "save_pretrained"):
            model.save_pretrained(str(kwargs["output_dir"]))
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Failed to apply LoRA: ")
        if kwargs.get("raise_on_error"):
            raise TrainingError(
                f"PyTorch PEFT failed: {e!s}",
                backend="pytorch",
                model_name=model_name,
                operation="apply_lora",
                cause=e,
            ) from e
        status = f"failed: {e!s}"
    return {
        "backend": "pytorch",
        "action": "apply_lora",
        "model": model_name,
        "target_modules": target_modules,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "status": status,
        "injected_modules": injected_count,
    }
