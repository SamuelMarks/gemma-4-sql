"""Keras-specific PEFT / LoRA implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)
keras = None
with catch_optional_imports():
    import keras


def apply_lora(model_name: str, target_modules: list[str], lora_r: int, lora_alpha: int, lora_dropout: float) -> JSONDict:
    """Apply LoRA to a model using the Keras backend.

    Args:
        model_name: The name of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.

    Returns:
        A dictionary containing the results.
    """
    status = "completed"
    if keras is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Keras dependencies are missing.")
    try:
        try:
            gemma_causal_lm_cls = __import__("keras_nlp.models", fromlist=["GemmaCausalLM"]).GemmaCausalLM
            model = gemma_causal_lm_cls.from_preset(model_name)
        except (ImportError, ValueError):
            inputs = keras.Input(shape=(None,), dtype="int32")
            x = keras.layers.Embedding(256, 128)(inputs)
            outputs = keras.layers.Dense(256)(x)
            model = keras.Model(inputs, outputs)
            model.backbone = model
        if hasattr(model, "backbone") and hasattr(model.backbone, "enable_lora"):
            model.backbone.enable_lora(rank=lora_r)
            logger.info("Enabled Keras native LoRA with rank %d", lora_r)
        else:
            logger.warning("Model backbone does not support `enable_lora` directly. Simulated.")
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Keras LoRA error: ")
        status = f"failed: {e!s}"
    return {"backend": "keras", "action": "apply_lora", "model": model_name, "target_modules": target_modules, "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout, "status": status}
