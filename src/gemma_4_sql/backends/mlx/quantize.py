"""MLX-specific model quantization logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_quantize import apply_bits_and_bytes_quantization, quantize_model_wrapper
from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)
mlx = None
BitsAndBytesConfig = None
AutoModelForCausalLM = None
with catch_optional_imports():
    import mlx
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # pragma: no cover


def quantize_model(model_name: str, method: str = "int8") -> JSONDict:
    """Quantize a MLX model.

    Args:
        model_name: The name of the target model.
        method: The string representing the method.

    Returns:
        A dictionary containing the results.
    """
    if mlx is None or BitsAndBytesConfig is None or AutoModelForCausalLM is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MLX dependencies are missing.")
    return quantize_model_wrapper(  # pragma: no cover
        backend_name="mlx",
        model_name=model_name,
        method=method,
        missing_deps=False,
        missing_status="",
        apply_fn=lambda: apply_bits_and_bytes_quantization(method, BitsAndBytesConfig, getattr(mlx, "float16", None)),
    )
