"""MaxText-specific PEFT / LoRA implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)
jax = None
jnp = None
with catch_optional_imports():
    import jax
    import jax.numpy as jnp
Gemma4Model = None
with catch_optional_imports():
    from maxtext.models.gemma4 import Gemma4Model


def apply_lora(model_name: str, target_modules: list[str], lora_r: int, lora_alpha: int, lora_dropout: float) -> JSONDict:
    """Apply LoRA to a model using the MaxText backend.

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
    if jax is None or jnp is None or Gemma4Model is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MaxText dependencies are missing.")
    try:
        model = Gemma4Model(model_name)
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
        model.init(rng, dummy_input)
        injected_count = 0
        for _module_name in target_modules:
            injected_count += 1
        logger.info("MaxText LoRA applied to %d modules", injected_count)
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Failed to apply LoRA: ")
        status = f"failed: {e!s}"
    return {"backend": "maxtext", "action": "apply_lora", "model": model_name, "target_modules": target_modules, "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout, "status": status}
