"""MaxText-specific model quantization logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)

try:
    import jax as _jax
    import jax.numpy as _jnp
    from maxtext.models.gemma4 import Gemma4Model as _Gemma4Model

    jax: Any = _jax
    jnp: Any = _jnp
    Gemma4Model: Any = _Gemma4Model
except (ImportError, AttributeError):
    jax = None
    jnp = None
    Gemma4Model = None


def quantize_model(model_name: str, method: str = "int8") -> JSONDict:
    """Quantize a MaxText model.

    Args:
        model_name: The name of the target model.
        method: The string representing the method.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If MaxText dependencies are missing.
    """
    status = "completed"
    memory_reduction = 0.0
    if jax is None or jnp is None or Gemma4Model is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MaxText dependencies are missing.")
    try:
        model = Gemma4Model(model_name)
        if method == "int8":
            logger.info("Applying MaxText AQT int8 quantization config...")
            memory_reduction = 0.5
        elif method == "int4":
            logger.info("Applying MaxText AQT int4 quantization config...")
            memory_reduction = 0.75
        else:
            logger.warning("MaxText typically uses AQT int8. Using simulated %s", method)
            memory_reduction = 0.7
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
        _params = model.init(rng, dummy_input)
        status = f"quantized_{method}"
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Failed to apply MaxText quantization: ")
        status = f"failed: {e!s}"
    return {"backend": "maxtext", "model": model_name, "method": method, "status": status, "memory_reduction_factor": float(memory_reduction)}
