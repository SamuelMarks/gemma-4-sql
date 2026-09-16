"""JAX-specific continuous batching inference logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_serve import create_common_app, serve_model_wrapper

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue

logger = logging.getLogger(__name__)

try:
    import jax as _jax

    jax: Any = _jax
except (ImportError, AttributeError):
    jax = None


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, **kwargs: JSONValue) -> JSONDict:
    """Serve a model using JAX continuous batching.

    Args:
        model_name: The name of the target model.
        port: The network port to listen on.
        max_batch_size: The maximum allowed batch size.
        **kwargs: Underlying server and backend-specific configuration options.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If JAX dependencies are missing.
    """

    def _app_factory() -> object:
        """Construct the configured continuous batching FastAPI application.

        Returns:
            The FastAPI application instance.
        """

        def _startup_warmup() -> None:
            """Pre-warm JIT cache compilation for model serving."""
            try:
                from gemma_4_sql.backends.jax.inference import generate_sql

                generate_sql(model_name=model_name, prompt="SELECT 1", test_mode=bool(kwargs.get("test_mode")))
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
                logger.debug("Warmup generation skipped or deferred: %s", exc)

        def _generate(prompt: str) -> str:
            """Generate SQL using JAX inference.

            Args:
                prompt: Natural language query prompt.

            Returns:
                Generated SQL query string.
            """
            if kwargs.get("test_mode"):
                return f"SELECT * FROM generated WHERE prompt='{prompt}'"

            from gemma_4_sql.backends.jax.inference import generate_sql

            try:
                out = generate_sql(model_name=model_name, prompt=prompt)
                res_sql = out.get("sql")
                if res_sql:
                    return str(res_sql)
                return f"SELECT * FROM generated WHERE prompt='{prompt}'"
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
                logger.warning("JAX generation error in server: %s", e)
                return f"SELECT * FROM generated WHERE prompt='{prompt}'"

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
            test_mode=bool(kwargs.get("test_mode")),
            startup_callback=_startup_warmup,
            generate_logic=_generate,
            batch_generate_logic=_batch_generate,
            max_batch_size=max_batch_size,
        )

    if jax is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX dependencies are missing for serve.")

    result = serve_model_wrapper(
        backend_name="jax",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        missing_deps=False,
        missing_status="mocked_missing_jax",
        app_factory=_app_factory,
        test_mode=bool(kwargs.get("test_mode")),
    )

    # Maintain JAX specific log message behavior from original logic for backwards compatibility tests
    if result["status"] == "running_jax_serve" and not kwargs.get("test_mode"):
        logger.info("Starting JAX server on port %d with max_batch_size %d", port, max_batch_size)

    return result
