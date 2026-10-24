"""JAX-specific continuous batching inference logic."""

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

        def _generate(prompt: str) -> str:
            """Generate SQL using JAX inference.

            Args:
                prompt: Natural language query prompt.

            Returns:
                Generated SQL query string.

            Raises:
                InferenceError: If JAX inference returns empty SQL or fails.

            """
            from gemma_4_sql.backends.jax.inference import generate_sql

            try:
                out = generate_sql(model_name=model_name, prompt=prompt)
                res_sql = out.get("sql", "")
                if res_sql:
                    return str(res_sql)
                raise InferenceError(f"JAX inference returned empty SQL for prompt '{prompt}'", backend="jax")
            except Exception as e:
                if isinstance(e, InferenceError):
                    raise
                logger.warning("JAX generation error in server: %s", e)
                raise InferenceError(f"JAX generation failed: {e}", backend="jax", cause=e) from e

        def _batch_generate(prompts: list[str]) -> list[str]:
            """Execute batched generation across multiple bundled prompt requests.

            Args:
                prompts: List of prompt strings.

            Returns:
                List of generated SQL queries.

            """
            return [_generate(p) for p in prompts]

        return create_common_app(
            backend_name="jax",
            model_name=model_name,
            startup_callback=_startup_warmup,
            generate_logic=_generate,
            batch_generate_logic=_batch_generate,
            max_batch_size=max_batch_size,
        )

    if jax is None:
        raise DependencyMissingError("JAX dependencies are missing for serve.", backend="jax")

    result = serve_model_wrapper(
        backend_name="jax",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        missing_deps=False,
        app_factory=_app_factory,
    )

    if result["status"] == "running_jax_serve":
        logger.info("Starting JAX server on port %d with max_batch_size %d", port, max_batch_size)

    return result
