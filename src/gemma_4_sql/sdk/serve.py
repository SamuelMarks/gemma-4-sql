"""SDK Serve module for continuous batching and vLLM inference."""

from __future__ import annotations

from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.type_hints import JSONValue, ServeResultDict


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, backend: str = "pytorch", **kwargs: JSONValue) -> ServeResultDict:
    """Serve a model using continuous batching.

    Args:
        model_name: The name of the target model.
        port: The network port to listen on.
        max_batch_size: The maximum allowed batch size.
        backend: The backend framework to use.
        **kwargs: Underlying server and backend-specific configuration options.

    Returns:
        Structured `ServeResultDict` containing `backend`, `model`, `port`, `max_batch_size`, `mode`, `status`, and FastAPI `app`.

    """
    return get_backend(backend).serve_model(model_name, port, max_batch_size, **kwargs)


__all__ = ["serve_model"]
