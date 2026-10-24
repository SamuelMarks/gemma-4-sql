"""PyTorch-specific PEFT / LoRA implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_peft import apply_lora_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import PEFTResultDict
logger = logging.getLogger(__name__)

peft = optional_import("peft")
torch = optional_import("torch")
LoraConfig = optional_attr("peft", "LoraConfig")
get_peft_model = optional_attr("peft", "get_peft_model")
AutoModelForCausalLM = optional_attr("transformers", "AutoModelForCausalLM")


def _count_injected_peft_modules(model: object, target_modules: list[str]) -> int:
    """Count injected LoRA adapter layers on a PyTorch PEFT model or fall back to `len(target_modules)`.

    Args:
        model: Wrapped PEFT model instance.
        target_modules: Target module name suffixes.

    Returns:
        Count of injected LoRA adapter modules.

    """
    named_mods_fn = getattr(model, "named_modules", None)
    if callable(named_mods_fn):
        try:
            mods_iter: Any = named_mods_fn()
            matched = sum(1 for name, mod in mods_iter if hasattr(mod, "lora_A") or hasattr(mod, "lora_a") or any(name.endswith(t) for t in target_modules))
            if matched > 0:
                return matched
        except (TypeError, ValueError, RuntimeError):
            pass
    return len(target_modules)


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

    def _inject() -> int:
        """Load base model, apply PEFT LoRA config, and optionally save adapters."""
        model = AutoModelForCausalLM.from_pretrained(model_name)
        lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        if "output_dir" in kwargs and hasattr(model, "save_pretrained"):
            model.save_pretrained(str(kwargs["output_dir"]))
        return _count_injected_peft_modules(model, target_modules)

    return apply_lora_wrapper(
        "pytorch",
        model_name,
        target_modules,
        lora_r,
        lora_alpha,
        lora_dropout,
        _inject,
        **kwargs,
    )
