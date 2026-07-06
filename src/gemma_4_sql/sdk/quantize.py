"""SDK interface for model quantization."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict


def quantize_model(model_name: str, method: str = "int8", backend: str = "pytorch") -> JSONDict:
    """Quantize a model using the specified method and backend.

    Args:
        model_name: The name of the target model.
        method: The string representing the method.
        backend: The backend framework to use.

    Returns:
        A dictionary containing the results.
    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).quantize_model(model_name, method)
