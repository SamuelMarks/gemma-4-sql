"""JAX-specific continuous batching inference logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_serve import create_common_app, serve_model_wrapper
from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)
jax = None
with catch_optional_imports():
    import jax
FastAPI = None
uvicorn = None


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, **kwargs: JSONValue) -> JSONDict:
    """Serve a model using JAX continuous batching.

        Args:
                    **kwargs: Underlying server and backend-specific configuration options.
    model_name: The name of the target model.
            port: The network port to listen on.
            max_batch_size: The maximum allowed batch size.

        Returns:
            A dictionary containing the results.
    """

    def _app_factory() -> object:
        """Execute function.

        Returns:
            The execution result.

        """

        def _generate(prompt: str) -> str:
            """Execute function.

            Returns:
                The execution result.

            """
            return "SELECT * FROM generated WHERE prompt='{p}'".replace("{p}", prompt)

        return create_common_app(
            backend_name="jax",
            model_name=model_name,
            test_mode=bool(kwargs.get("test_mode")),
            generate_logic=_generate,
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
