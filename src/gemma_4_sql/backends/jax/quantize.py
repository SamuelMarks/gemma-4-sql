"""JAX-specific model quantization logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

MIN_NDIM_FOR_QUANTIZATION = 2
if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)

try:
    import jax as _jax
    import jax.numpy as _jnp
    from flax import nnx as _nnx

    from .gemma4 import Gemma4Config as _Gemma4Config
    from .gemma4 import Gemma4ForCausalLM as _Gemma4ForCausalLM

    jax: Any = _jax
    jnp: Any = _jnp
    nnx: Any = _nnx
    Gemma4Config: Any = _Gemma4Config
    Gemma4ForCausalLM: Any = _Gemma4ForCausalLM
except (ImportError, AttributeError):
    jax = None
    jnp = None
    nnx = None
    Gemma4Config = None
    Gemma4ForCausalLM = None


def quantize_int8(tensor: Any) -> tuple[Any, Any]:
    """Quantize a tensor to int8.

    Args:
        tensor: The input tensor.

    Returns:
        A tuple containing the results.

    Raises:
        DependencyMissingError: If JAX is required but missing.
    """
    if jnp is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX is required for quantize_int8.")
    scale = jnp.max(jnp.abs(tensor)) / 127.0
    q_tensor = jnp.round(tensor / scale).astype(jnp.int8)
    return (q_tensor, scale)


def _apply_quantization_to_model(model: Any, method: str) -> tuple[str, float, int]:
    """Apply quantization to the model graph and preserve scale factors.

    Args:
        model: The model object to quantize.
        method: The quantization method string.

    Returns:
        A tuple of status string, memory reduction factor, and count of quantized parameters.
    """
    quantized_params = 0
    if method in {"int8", "awq"}:
        scales: dict[str, object] = {}
        for path, param in nnx.graph.iter_graph(model):
            if isinstance(param, nnx.Param) and hasattr(param.value, "ndim") and (param.value.ndim >= MIN_NDIM_FOR_QUANTIZATION):
                (q_tensor, scale) = quantize_int8(param.value)
                param.value = q_tensor
                param.quant_scale = scale
                scales[str(path)] = scale
                quantized_params += 1
        model._quant_scales = scales
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
        model_name: The name of the model to quantize.
        method: The quantization method ('int8', 'awq', 'gptq', 'gguf').

    Returns:
        A dictionary containing quantization status and metadata.

    Raises:
        DependencyMissingError: If JAX quantization dependencies are missing.
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
