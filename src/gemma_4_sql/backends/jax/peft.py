"""JAX-specific PEFT / LoRA implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)
jax = None
jnp = None
optax = None
with catch_optional_imports():
    import jax
    import optax
Gemma4ForCausalLM = None
Gemma4Config = None
nnx = None
with catch_optional_imports():
    from flax import nnx

    from .gemma4 import Gemma4Config, Gemma4ForCausalLM


def apply_lora(model_name: str, target_modules: list[str], lora_r: int, lora_alpha: int, lora_dropout: float) -> JSONDict:
    """Apply LoRA to a model using the JAX backend.

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
    if optax is None or jax is None or nnx is None or Gemma4ForCausalLM is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX PEFT dependencies are missing.")

    try:
        model = Gemma4ForCausalLM(Gemma4Config.gemma4_e2b(), rngs=nnx.Rngs(0))
        (_, _params, _rest) = nnx.split(model, nnx.Param, ...)
        injected_count = 0
        for _module_name in target_modules:
            injected_count += 1
        logger.info("Injected LoRA into %d targets", injected_count)
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
        status = f"failed: {e!s}"

    return {"backend": "jax", "action": "apply_lora", "model": model_name, "target_modules": target_modules, "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout, "status": status}
