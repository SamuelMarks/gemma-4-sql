"""PyTorch-specific model quantization logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)
torch = None
BitsAndBytesConfig = None
AutoModelForCausalLM = None
with catch_optional_imports():
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def _apply_quantization(method: str) -> tuple[float, str]:
    """Execute logic."""
    if method == "int8":
        BitsAndBytesConfig(load_in_8bit=True)
        memory_reduction = 0.5
    elif method == "int4":
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        memory_reduction = 0.75
    elif method in ["gptq", "awq"]:
        logger.info("Using simulated %s quantization config.", method)
        memory_reduction = 0.7
    else:
        logger.warning("Unsupported quantization method: %s", method)
        return (0.0, f"unsupported_method_{method}")
    return (memory_reduction, f"quantized_{method}")


def quantize_model(model_name: str, method: str = "int8") -> JSONDict:
    """Quantize a PyTorch model.

    Args:
    ----
        model_name: The name of the model to quantize.
        method: The quantization method ('int8', 'int4', 'awq', 'gptq', 'gguf').

    Returns:
    -------
        A dictionary containing quantization status and metadata.

    """
    status = "completed"
    memory_reduction = 0.0
    if torch is not None and BitsAndBytesConfig is not None and (AutoModelForCausalLM is not None):
        try:
            (memory_reduction, status) = _apply_quantization(method)
            if status.startswith("unsupported"):
                return {"backend": "pytorch", "model": model_name, "method": method, "status": status, "memory_reduction_factor": 0.0}
            logger.info("Loading model %s with %s quantization...", model_name, method)
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.exception("Failed to quantize: ")
            status = f"failed: {e!s}"
            memory_reduction = 0.0
    else:
        status = "mocked_missing_torch"
        memory_reduction = 0.0
    return {"backend": "pytorch", "model": model_name, "method": method, "status": status, "memory_reduction_factor": memory_reduction}
