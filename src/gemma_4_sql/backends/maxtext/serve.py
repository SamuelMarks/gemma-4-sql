"""MaxText-specific continuous batching inference logic."""

from __future__ import annotations

import logging

from gemma_4_sql.backends.common_serve import create_backend_serve_app, serve_model_wrapper
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import FastAPIAppProtocol, JSONDict, JSONValue, ServeResultDict

logger = logging.getLogger(__name__)

jax = optional_import("jax")
gemma4 = optional_import("maxtext.models.gemma4")


def _create_app(model_name: str, max_batch_size: int = 32) -> FastAPIAppProtocol:
    """Create the FastAPI application for the MaxText server.

    Args:
        model_name: The name of the target model.
        max_batch_size: Maximum batch size per continuous batching window.

    Returns:
        The FastAPI application instance.

    """

    def _startup() -> None:
        """Initialize JAX distributed runtime on FastAPI startup."""
        try:
            assert jax is not None
            jax.distributed.initialize()
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError, AssertionError) as e:
            logger.warning("jax.distributed.initialize() failed: %s", e)

    def _call_maxtext_generate(model_name: str, prompt: str) -> JSONDict:
        """Delegate single-prompt generation to `maxtext.inference.generate_sql`."""
        from gemma_4_sql.backends.maxtext.inference import generate_sql

        res = generate_sql(model_name=model_name, prompt=prompt)
        return {k: v for k, v in res.items() if isinstance(v, (str, int, float, bool)) or v is None}

    return create_backend_serve_app(
        backend_name="maxtext",
        model_name=model_name,
        generate_sql_fn=_call_maxtext_generate,
        startup_callback=_startup,
        max_batch_size=max_batch_size,
    )


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, **_kwargs: JSONValue) -> ServeResultDict:
    """Serve a model using MaxText continuous batching.

    Args:
        model_name: The name of the target model.
        port: The network port to listen on.
        max_batch_size: The maximum allowed batch size.
        **_kwargs: Underlying server and backend-specific configuration options.

    Returns:
        A structured `ServeResultDict` containing the server metadata.

    Raises:
        DependencyMissingError: If MaxText dependencies are missing.

    """
    if gemma4 is None or jax is None:
        raise DependencyMissingError("MaxText dependencies are missing.", backend="maxtext")
    return serve_model_wrapper(
        backend_name="maxtext",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        app_factory=lambda: _create_app(model_name, max_batch_size=max_batch_size),
    )
