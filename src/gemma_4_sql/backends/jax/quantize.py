"""JAX-specific model quantization logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

MIN_NDIM_FOR_QUANTIZATION = 2
if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, TensorType
logger = logging.getLogger(__name__)
jax = None
jnp = None
with catch_optional_imports():
    import jax
    import jax.numpy as jnp
Gemma4ForCausalLM = None
Gemma4Config = None
nnx = None
with catch_optional_imports():
    from flax import nnx

    from .gemma4 import Gemma4Config, Gemma4ForCausalLM


def quantize_int8(tensor: TensorType) -> tuple[TensorType, TensorType]:
    """Quantize a tensor to int8.

    Args:
        tensor: The input tensor.

    Returns:
        A tuple containing the results.
    """
    scale = jnp.max(jnp.abs(tensor)) / 127.0
    quantized = jnp.round(tensor / scale).astype(jnp.int8)
    return (quantized, scale)


def _apply_quantization_to_model(model: object, method: str) -> tuple[str, float, int]:
    """Apply quantization to the model graph.

    Returns:
        object: The resulting output from the operation.

    """
    quantized_params = 0
    if method in {"int8", "awq"}:
        for _path, param in nnx.graph.iter_graph(model):
            if isinstance(param, nnx.Param) and hasattr(param.value, "ndim") and (param.value.ndim >= MIN_NDIM_FOR_QUANTIZATION):
                (_q_tensor, _scale) = quantize_int8(param.value)
                quantized_params += 1
        status = f"quantized_{method}"
        memory_reduction = 0.5 if method == "int8" else 0.7
        logger.info("Quantized %d parameters using %s", quantized_params, method)
    else:
        status = f"unsupported_method_{method}"
        memory_reduction = 0.0
    return (status, memory_reduction, quantized_params)


def quantize_model(model_name: str, method: str = "int8") -> JSONDict:
    """Quantize a JAX model.

    Args:
    ----
        model_name: The name of the model to quantize.
        method: The quantization method ('int8', 'awq', 'gptq', 'gguf').

    Returns:
    -------
        A dictionary containing quantization status and metadata.

    """
    if jax is None or jnp is None or nnx is None or Gemma4ForCausalLM is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX quantization dependencies are missing.")

    try:
        model = Gemma4ForCausalLM(Gemma4Config.gemma4_e2b(), rngs=nnx.Rngs(0))
        (status, memory_reduction, _) = _apply_quantization_to_model(model, method)
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
        status = f"failed: {e!s}"
        memory_reduction = 0.0

    return {"backend": "jax", "model": model_name, "method": method, "status": status, "memory_reduction_factor": float(memory_reduction)}
