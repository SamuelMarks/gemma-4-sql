"""SDK interface for model quantization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.sdk.registry import get_backend

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONValue, QuantizeResultDict


def quantize_model(model_name: str, method: str = "int8", backend: str = "pytorch", **kwargs: JSONValue) -> QuantizeResultDict:
    """Quantize a model using the specified quantization scheme and backend framework.

    Args:
        model_name: Hugging Face model identifier or local checkpoint path.
        method: Quantization scheme (`'int8'`, `'int4'`, `'awq'`, `'gptq'`, or `'gguf'`).
        backend: Target backend framework (`'pytorch'`, `'jax'`, `'keras'`, `'maxtext'`, or `'mlx'`).
        **kwargs: Additional backend-specific quantization options (e.g., `export_path`, `calibration_data`).

    Returns:
        Structured `QuantizeResultDict` containing `backend`, `model`, `method`, `status`, and `memory_reduction_factor`.

    """
    return get_backend(backend).quantize_model(model_name, method, **kwargs)


__all__ = ["quantize_model"]
