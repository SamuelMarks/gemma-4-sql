"""JAX-specific continuous batching inference logic."""

from __future__ import annotations

import logging

from gemma_4_sql.backends.common_serve import create_backend_serve_app, serve_model_wrapper
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import FastAPIAppProtocol, JSONDict, JSONValue, ServeResultDict

logger = logging.getLogger(__name__)

jax = optional_import("jax")


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, **_kwargs: JSONValue) -> ServeResultDict:
    """Serve a model using JAX continuous batching.

    Args:
        model_name: The name of the target model.
        port: The network port to listen on.
        max_batch_size: The maximum allowed batch size.
        **_kwargs: Underlying server and backend-specific configuration options.

    Returns:
        A structured `ServeResultDict` containing the server metadata.

    Raises:
        DependencyMissingError: If JAX dependencies are missing.

    """

    def _app_factory() -> FastAPIAppProtocol:
        """Construct the configured continuous batching FastAPI application.

        Returns:
            The FastAPI application instance.

        """

        def _startup_warmup() -> None:
            """Pre-warm JIT cache compilation for model serving."""
            try:
                from gemma_4_sql.backends.jax.inference import generate_sql

                generate_sql(model_name=model_name, prompt="SELECT 1")
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
                logger.debug("Warmup generation skipped or deferred: %s", exc)

        def _call_jax_generate(model_name: str, prompt: str) -> JSONDict:
            """Delegate single-prompt generation to `jax.inference.generate_sql`."""
            from gemma_4_sql.backends.jax.inference import generate_sql

            res = generate_sql(model_name=model_name, prompt=prompt)
            return {k: v for k, v in res.items() if isinstance(v, (str, int, float, bool)) or v is None}

        return create_backend_serve_app(
            backend_name="jax",
            model_name=model_name,
            generate_sql_fn=_call_jax_generate,
            startup_callback=_startup_warmup,
            max_batch_size=max_batch_size,
        )

    if jax is None:
        raise DependencyMissingError("JAX dependencies are missing for serve.", backend="jax")

    result = serve_model_wrapper(
        backend_name="jax",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        app_factory=_app_factory,
    )

    if result["status"] == "running_jax_serve":
        logger.info("Starting JAX server on port %d with max_batch_size %d", port, max_batch_size)

    return result
