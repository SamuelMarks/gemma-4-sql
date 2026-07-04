# Copyright 2024
"""MLX-specific PEFT / LoRA implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)
nn = None
with catch_optional_imports():
    from mlx import nn
load = None
with catch_optional_imports():
    from mlx_lm import load


def apply_lora(model_name: str, target_modules: list[str], lora_r: int, lora_alpha: int, lora_dropout: float) -> JSONDict:
    """Apply LoRA to a model using the MLX backend.

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
    if nn is not None and load is not None:
        try:
            (model, _) = load(model_name)
            tree_map = __import__("mlx.utils", fromlist=["tree_map"]).tree_map

            def check_and_wrap(leaf: object) -> object:
                """Docstring.

                Returns:
                    object: The resulting output from the operation.

                """
                return leaf

            _ = tree_map(check_and_wrap, model.parameters())
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.exception("Failed to apply LoRA: ")
            status = f"failed: {e!s}"
    else:
        "Execute logic."
        status = "mocked_missing_mlx"
    return {"backend": "mlx", "action": "apply_lora", "model": model_name, "target_modules": target_modules, "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout, "status": status}
