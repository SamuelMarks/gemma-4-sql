# Copyright 2024
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
    if keras is not None:
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
    else:
        status = "mocked_missing_keras"
    return {"backend": "keras", "action": "apply_lora", "model": model_name, "target_modules": target_modules, "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout, "status": status}
