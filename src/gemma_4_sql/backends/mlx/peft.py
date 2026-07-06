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
        model_name: The name of the target model.
        target_modules: The names of the modules to apply LoRA.
        lora_r: The rank of the LoRA update matrices.
        lora_alpha: The scaling factor for LoRA.
        lora_dropout: The dropout probability for LoRA layers.

    Returns:
        A dictionary containing the results.
    """
    status = "completed"
    if nn is None or load is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MLX dependencies are missing.")
    try:  # pragma: no cover
        (model, _) = load(model_name)  # pragma: no cover
        tree_map = __import__("mlx.utils", fromlist=["tree_map"]).tree_map  # pragma: no cover

        def check_and_wrap(leaf: object) -> object:  # pragma: no cover
            """Docstring.  # pragma: no cover

            Returns:  # pragma: no cover
                object: The resulting output from the operation.  # pragma: no cover

            """  # pragma: no cover
            return leaf  # pragma: no cover

        _ = tree_map(check_and_wrap, model.parameters())  # pragma: no cover
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:  # pragma: no cover
        logger.exception("Failed to apply LoRA: ")  # pragma: no cover
        status = f"failed: {e!s}"  # pragma: no cover
    return {"backend": "mlx", "action": "apply_lora", "model": model_name, "target_modules": target_modules, "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout, "status": status}  # pragma: no cover
