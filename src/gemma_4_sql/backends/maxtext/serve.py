"""MaxText-specific continuous batching inference logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_serve import create_common_app, serve_model_wrapper
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import FastAPIAppProtocol, JSONValue, ServeResultDict
logger = logging.getLogger(__name__)

jax = optional_import("jax")
gemma4 = optional_import("maxtext.models.gemma4")


def _create_app(model_name: str) -> FastAPIAppProtocol:
    """Create the FastAPI application for the MaxText server.

    Args:
        model_name: The name of the target model.

    Returns:
        The FastAPI application instance.

    """

    def _startup() -> None:
        """Initialize JAX distributed runtime on FastAPI startup."""
        try:
            jax.distributed.initialize()
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.warning("jax.distributed.initialize() failed: %s", e)

    def _generate(prompt: str) -> str:
        """Generate SQL using MaxText inference.

        Args:
            prompt: Natural language query prompt.

        Returns:
            Generated SQL query string.

        Raises:
            InferenceError: If model inference fails during execution.

        """
        from gemma_4_sql.backends.maxtext.inference import generate_sql

        try:
            out = generate_sql(model_name=model_name, prompt=prompt)
            sql = out.get("sql", "")
            if sql:
                return str(sql)
            raise InferenceError(f"MaxText inference returned empty SQL for prompt '{prompt}'", backend="maxtext")
        except Exception as e:
            if isinstance(e, InferenceError):
                raise
            raise InferenceError(f"MaxText generation failed: {e}", backend="maxtext", cause=e) from e

    return create_common_app(
        backend_name="maxtext",
        model_name=model_name,
        startup_callback=_startup,
        generate_logic=_generate,
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
        missing_deps=False,
        app_factory=lambda: _create_app(model_name),
    )
