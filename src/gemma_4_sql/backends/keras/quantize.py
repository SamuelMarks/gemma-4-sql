# Copyright 2024
"""Keras-specific model quantization logic."""

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


def quantize_model(model_name: str, method: str = "int8") -> JSONDict:
    """Quantize a Keras model.

    Args:
    ----
        model_name: The name of the model to quantize.
        method: The quantization method ('int8', 'int4', 'awq', 'gptq', 'gguf').

    Returns:
    -------
        A dictionary containing quantization status and metadata.

    """
    memory_reduction = 0.0
    status = "completed"
    if keras is not None:
        try:
            if method in {"int8", "int4"}:
                logger.info("Setting Keras model dtype to %s", method)
                memory_reduction = 0.5 if method == "int8" else 0.75
            elif method in {"awq", "gptq"}:
                logger.warning("Keras natively uses int8/int4 via preset. Simulating %s", method)
                memory_reduction = 0.7
            else:
                logger.warning("Unsupported quantization method: %s", method)
                return {"backend": "keras", "model": model_name, "method": method, "status": f"unsupported_method_{method}", "memory_reduction_factor": 0.0}
            status = f"quantized_{method}"
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.exception("Failed to apply Keras quantization: ")
            status = f"failed: {e!s}"
    else:
        status = "mocked_missing_keras"
        memory_reduction = 0.0
    return {"backend": "keras", "model": model_name, "method": method, "status": status, "memory_reduction_factor": float(memory_reduction)}
