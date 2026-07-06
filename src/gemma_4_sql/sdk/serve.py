"""SDK Serve module for continuous batching and vLLM inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, backend: str = "pytorch", **kwargs: JSONValue) -> JSONDict:
    """Serve a model using continuous batching.

        Args:
                    **kwargs: Underlying server and backend-specific configuration options.
    model_name: The name of the target model.
            port: The network port to listen on.
            max_batch_size: The maximum allowed batch size.
            backend: The backend framework to use.

        Returns:
            A dictionary containing the results.
    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).serve_model(model_name, port, max_batch_size, **kwargs)
