"""Parameter-Efficient Fine-Tuning (PEFT / LoRA) configuration module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue


def apply_peft(model_name: str, target_modules: list[str] | None = None, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05, **kwargs: JSONValue) -> JSONDict:
    """Apply Parameter-Efficient Fine-Tuning (PEFT / LoRA) to a model.

    Args:
    ----
        model_name: The name of the model to fine-tune.
        target_modules: List of target modules for LoRA. Defaults to ["q_proj", "v_proj"].
        lora_r: LoRA attention dimension (rank).
        lora_alpha: LoRA alpha parameter.
        lora_dropout: LoRA dropout probability.
        backend: The backend approach to use ('jax', 'keras', 'maxtext', 'pytorch').
        **kwargs: Additional keyword arguments.

    Returns:
    -------
        A dictionary indicating the PEFT job status and configuration.

    """
    backend = kwargs.get("backend", "jax")
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).apply_lora(model_name, target_modules, lora_r, lora_alpha, lora_dropout)
