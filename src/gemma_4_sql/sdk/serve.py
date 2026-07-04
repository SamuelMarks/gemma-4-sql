# Copyright 2024
"""SDK Serve module for continuous batching and vLLM inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, backend: str = "pytorch", **kwargs: JSONValue) -> JSONDict:
    """Serve a model using continuous batching.

    Args:
    ----
        model_name: The name of the model to serve.
        port: The port to bind the server to.
        max_batch_size: The maximum batch size.
        backend: The backend framework ('jax', 'keras', 'maxtext', 'pytorch').
        **kwargs: Additional parameters.

    Returns:
    -------
        Serving configuration and status dictionary.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).serve_model(model_name, port, max_batch_size, **kwargs)
