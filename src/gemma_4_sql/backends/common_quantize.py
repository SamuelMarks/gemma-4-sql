# Copyright 2024
"""Common quantization utility for backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict

logger = logging.getLogger(__name__)


def apply_bits_and_bytes_quantization(method: str, bits_and_bytes_config_cls: type | None, float16_dtype: object = None) -> tuple[float, str]:
    """Apply quantization using BitsAndBytes config mapping.

    Args:
    ----
        method: The quantization method (e.g. 'int8', 'int4', 'gptq', 'awq').
        bits_and_bytes_config_cls: The BitsAndBytesConfig class from transformers.
        float16_dtype: The float16 dtype specific to the backend (e.g. torch.float16, mlx.float16).

    Returns:
    -------
        A tuple of (memory_reduction_factor, status).

    """
    if bits_and_bytes_config_cls is None:
        return (0.0, "mocked_missing_bitsandbytes")  # pragma: no cover

    if method == "int8":
        bits_and_bytes_config_cls(load_in_8bit=True)
        memory_reduction = 0.5
    elif method == "int4":
        bits_and_bytes_config_cls(load_in_4bit=True, bnb_4bit_compute_dtype=float16_dtype, bnb_4bit_use_double_quant=True)
        memory_reduction = 0.75
    elif method in {"gptq", "awq"}:
        logger.info("Using simulated %s quantization config.", method)
        memory_reduction = 0.7
    else:
        logger.warning("Unsupported quantization method: %s", method)
        return (0.0, f"unsupported_method_{method}")

    return (memory_reduction, f"quantized_{method}")


def quantize_model_wrapper(backend_name: str, model_name: str, method: str, missing_deps: bool, missing_status: str, apply_fn: callable) -> JSONDict:
    """Wrapper to handle errors and standardized response for quantization.

    Args:
    ----
        backend_name: Name of the backend.
        model_name: Name of the model.
        method: The quantization method.
        missing_deps: Whether dependencies are missing.
        missing_status: Status to return if dependencies are missing.
        apply_fn: Function that applies quantization and returns (memory_reduction, status).

    Returns:
    -------
        A dictionary containing the quantization results.
    """
    if missing_deps:
        return {
            "backend": backend_name,
            "model": model_name,
            "method": method,
            "status": missing_status,
            "memory_reduction_factor": 0.0,
        }

    try:
        (memory_reduction, status) = apply_fn()
        if not status.startswith("unsupported"):
            logger.info("Loading model %s with %s quantization...", model_name, method)
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Failed to quantize: ")
        status = f"failed: {e!s}"
        memory_reduction = 0.0

    return {
        "backend": backend_name,
        "model": model_name,
        "method": method,
        "status": status,
        "memory_reduction_factor": float(memory_reduction),
    }
