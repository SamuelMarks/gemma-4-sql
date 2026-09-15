"""MLX-specific continuous batching inference logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_serve import create_common_app, serve_model_wrapper

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue

logger = logging.getLogger(__name__)

try:
    import mlx.core as _mx

    mx: Any = _mx
except (ImportError, AttributeError):
    mx = None


def _generate_query(prompt: str) -> str:
    """Generate query for prompt.

    Args:
        prompt: Natural language input prompt.

    Returns:
        Generated SQL query string.
    """
    return f"SELECT * FROM generated WHERE prompt='{prompt}'"


def _app_factory(model_name: str, test_mode: bool = False) -> object:
    """Construct FastAPI app for MLX model serving.

    Args:
        model_name: The name of the model being served.
        test_mode: Whether running in test mode.

    Returns:
        The FastAPI application instance.
    """
    return create_common_app(
        backend_name="mlx",
        model_name=model_name,
        test_mode=test_mode,
        generate_logic=_generate_query,
    )


def serve_model(
    model_name: str,
    port: int = 8000,
    max_batch_size: int = 256,
    **kwargs: JSONValue,
) -> JSONDict:
    """Serve a model using MLX continuous batching.

    Args:
        model_name: The name of the target model.
        port: The network port to listen on.
        max_batch_size: The maximum allowed batch size.
        **kwargs: Underlying server and backend-specific configuration options.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If MLX dependencies are not installed.
    """
    if mx is None and not kwargs.get("test_mode"):
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MLX dependencies are missing for serve.")

    result = serve_model_wrapper(
        backend_name="mlx",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        missing_deps=mx is None and not bool(kwargs.get("test_mode")),
        missing_status="mocked_missing_mlx",
        app_factory=lambda: _app_factory(model_name, bool(kwargs.get("test_mode"))),
        test_mode=bool(kwargs.get("test_mode")),
    )

    if result["status"] == "running_mlx_serve" and not kwargs.get("test_mode"):
        logger.info("Starting MLX server on port %d with max_batch_size %d", port, max_batch_size)

    return result
