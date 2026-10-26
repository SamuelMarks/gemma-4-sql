"""Parameter-Efficient Fine-Tuning (PEFT / LoRA) configuration module."""

from __future__ import annotations

from gemma_4_sql.constants import DEFAULT_BACKEND
from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.type_hints import JSONValue, PEFTResultDict


def apply_peft(
    model_name: str,
    target_modules: list[str] | None = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    backend: str = DEFAULT_BACKEND,
    **kwargs: JSONValue,
) -> PEFTResultDict:
    """Apply Parameter-Efficient Fine-Tuning (PEFT / LoRA) to a model.

    Args:
        model_name: The name of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.
        backend: Target backend name (defaults to `DEFAULT_BACKEND`).
        **kwargs: Backend-specific LoRA parameters.

    Returns:
        Structured `PEFTResultDict` containing `backend`, `action`, `model`, `target_modules`, LoRA hyperparameters, and `status`.

    """
    resolved_modules = list(target_modules) if target_modules is not None else ["q_proj", "v_proj"]

    return get_backend(backend).apply_lora(
        model_name,
        resolved_modules,
        lora_r,
        lora_alpha,
        lora_dropout,
        **kwargs,
    )


__all__ = ["apply_peft"]
