"""Parameter-Efficient Fine-Tuning (PEFT / LoRA) configuration module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.sdk.registry import get_backend

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONValue, PEFTResultDict


def apply_peft(
    model_name: str,
    target_modules: list[str] | None = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    **kwargs: JSONValue,
) -> PEFTResultDict:
    """Apply Parameter-Efficient Fine-Tuning (PEFT / LoRA) to a model.

    Args:
        model_name: The name of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.
        **kwargs: Backend-specific LoRA parameters (including optional `backend`).

    Returns:
        Structured `PEFTResultDict` containing `backend`, `action`, `model`, `target_modules`, LoRA hyperparameters, and `status`.

    """
    call_kwargs = dict(kwargs)
    backend = str(call_kwargs.pop("backend", "jax"))
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    return get_backend(backend).apply_lora(
        model_name,
        target_modules,
        lora_r,
        lora_alpha,
        lora_dropout,
        **call_kwargs,
    )


__all__ = ["apply_peft"]
