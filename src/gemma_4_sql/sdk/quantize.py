"""
SDK interface for model quantization.
"""

from __future__ import annotations

from typing import Any


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
    from gemma_4_sql.sdk.registry import get_backend

    return get_backend(backend).quantize_model(model_name, method, **kwargs)
