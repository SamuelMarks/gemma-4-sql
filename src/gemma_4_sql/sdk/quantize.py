"""
SDK interface for model quantization.
"""

from __future__ import annotations

from typing import Any

from gemma_4_sql.backends.jax.quantize import quantize_model as quantize_jax
from gemma_4_sql.backends.keras.quantize import quantize_model as quantize_keras
from gemma_4_sql.backends.maxtext.quantize import quantize_model as quantize_maxtext
from gemma_4_sql.backends.pytorch.quantize import quantize_model as quantize_pytorch


def quantize_model(
    model_name: str, method: str = "int8", backend: str = "pytorch", **kwargs: Any
) -> dict[str, Any]:
    """
    Quantizes a model using the specified method and backend.

    Args:
        model_name: Name of the model to quantize.
        method: The quantization method ('int8', 'awq', 'gptq', 'gguf').
        backend: The execution backend ('jax', 'keras', 'maxtext', 'pytorch').
        **kwargs: Additional quantization parameters.

    Returns:
        A dict with execution status and metrics.

    Raises:
        ValueError: If an unknown backend is provided.
    """
    if backend == "pytorch":
        return quantize_pytorch(model_name, method, **kwargs)
    elif backend == "jax":
        return quantize_jax(model_name, method, **kwargs)
    elif backend == "keras":
        return quantize_keras(model_name, method, **kwargs)
    elif backend == "maxtext":
        return quantize_maxtext(model_name, method, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
