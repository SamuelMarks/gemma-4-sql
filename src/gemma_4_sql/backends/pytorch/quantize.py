"""PyTorch-specific model quantization logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_quantize import apply_bits_and_bytes_quantization, quantize_model_wrapper
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


def quantize_model(model_name: str, method: str = "int8") -> JSONDict:
    """Quantize a PyTorch model.

    Args:
        model_name: The name of the target model.
        method: The string representing the method.

    Returns:
        A dictionary containing the results.
    """
    if torch is None or BitsAndBytesConfig is None or AutoModelForCausalLM is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("PyTorch quantization dependencies are missing.")

    return quantize_model_wrapper(
        backend_name="pytorch",
        model_name=model_name,
        method=method,
        missing_deps=False,
        missing_status="mocked_missing_torch",
        apply_fn=lambda: apply_bits_and_bytes_quantization(method, BitsAndBytesConfig, getattr(torch, "float16", None)),
    )
