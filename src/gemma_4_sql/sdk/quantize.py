"""SDK interface for model quantization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.sdk.registry import get_backend

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONValue, QuantizeResultDict


def quantize_model(model_name: str, method: str = "int8", backend: str = "pytorch", **kwargs: JSONValue) -> QuantizeResultDict:
    """Quantize a model using the specified method and backend.

    Args:
        model_name: The name of the target model.
        method: The string representing the method.
        backend: The backend framework to use.
        **kwargs: Additional backend-specific quantization options (e.g., export_path, calibration_data).

    Returns:
        A dictionary containing the results.

    """
    return get_backend(backend).quantize_model(model_name, method, **kwargs)


__all__ = ["quantize_model"]
