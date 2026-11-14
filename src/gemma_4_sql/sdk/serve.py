"""SDK Serve module for continuous batching and vLLM inference."""

from __future__ import annotations

from gemma_4_sql.backends.common_serve import get_serve_handle
from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.type_hints import JSONValue, ServeHandle, ServeResultDict


def serve_model(
    model_name: str,
    port: int = 8000,
    max_batch_size: int = 256,
    backend: str = "pytorch",
    **kwargs: JSONValue,
) -> ServeResultDict:
    """Serve a model using continuous batching and return JSON-serializable metadata.

    Args:
        model_name: Name or path of the target model.
        port: Network TCP port to listen on.
        max_batch_size: Maximum allowed continuous batch size.
        backend: Backend framework identifier.
        **kwargs: Backend-specific server configuration options.

    Returns:
        JSON-serializable `ServeResultDict` containing `backend`, `model`, `port`,
        `max_batch_size`, `mode`, and `status`.

    """
    return get_backend(backend).serve_model(model_name, port, max_batch_size, **kwargs)


def serve_model_handle(
    model_name: str,
    port: int = 8000,
    max_batch_size: int = 256,
    backend: str = "pytorch",
    **kwargs: JSONValue,
) -> ServeHandle:
    """Initialize a continuous batching server and return a `ServeHandle` with both metadata and live ASGI app.

    Args:
        model_name: Name or path of the target model.
        port: Network TCP port to listen on.
        max_batch_size: Maximum allowed continuous batch size.
        backend: Backend framework identifier.
        **kwargs: Backend-specific server configuration options.

    Returns:
        `ServeHandle` pairing JSON-serializable `ServeResultDict` metadata with the live `FastAPIAppProtocol` instance.

    """
    metadata = serve_model(model_name, port=port, max_batch_size=max_batch_size, backend=backend, **kwargs)
    return get_serve_handle(metadata)


__all__ = ["get_serve_handle", "serve_model", "serve_model_handle"]
