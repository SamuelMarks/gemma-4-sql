"""MLX-specific continuous batching inference logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from gemma_4_sql.backends.common_serve import create_common_app, serve_model_wrapper
from gemma_4_sql.type_hints import ModelType

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue

logger = logging.getLogger(__name__)

try:
    import mlx.core as _mx

    mx: Any = _mx
except (ImportError, AttributeError):
    mx = None

_mlx_model_cache: dict[str, tuple[Any, Any]] = {}


def _load_mlx_model(model_name: str) -> tuple[Any, Any]:
    """Load an MLX model and tokenizer into cache.

    Args:
        model_name: The name or path of the target MLX model.

    Returns:
        A tuple of (model, tokenizer).

    Raises:
        DependencyMissingError: If mlx_lm dependencies are missing.

    """
    if model_name in _mlx_model_cache:
        return _mlx_model_cache[model_name]
    try:
        import mlx_lm

        loaded: Any = mlx_lm.load(model_name)
        if isinstance(loaded, (tuple, list)):
            model: ModelType = cast(list[Any], loaded)[0]
            tokenizer: Any = cast(list[Any], loaded)[1] if len(cast(list[Any], loaded)) > 1 else None
        else:
            model = loaded
            tokenizer = None
        _mlx_model_cache[model_name] = (model, tokenizer)
        return model, tokenizer
    except (ImportError, ValueError, OSError, RuntimeError, AttributeError) as err:
        logger.warning("Could not load MLX model %s: %s", model_name, err)
        raise


def _generate_query(prompt: str, model_name: str = "") -> str:
    """Generate a SQL query for a prompt using MLX inference.

    Args:
        prompt: Natural language input prompt.
        model_name: Target model identifier.

    Returns:
        Generated SQL query string.

    Raises:
        InferenceError: If model inference fails during non-test execution.

    """
    from gemma_4_sql.backends.mlx.inference import generate_sql
    from gemma_4_sql.exceptions import InferenceError

    try:
        res = generate_sql(model_name=model_name or "default", prompt=prompt)
        sql = res.get("sql", "")
        if sql:
            return str(sql)
        raise InferenceError(f"MLX inference returned empty SQL for prompt '{prompt}'")
    except Exception as e:
        if isinstance(e, InferenceError):
            raise
        logger.error("MLX serve generation encountered error: %s", e)
        raise InferenceError(f"MLX generation failed: {e}") from e


def _batch_generate_queries(prompts: list[str], model_name: str = "") -> list[str]:
    """Generate SQL queries for a batch of prompts using MLX inference.

    Args:
        prompts: Sequence of natural language prompts.
        model_name: Target model identifier.

    Returns:
        List of generated SQL query strings.

    """
    return [_generate_query(p, model_name=model_name) for p in prompts]


def _app_factory(model_name: str) -> object:
    """Construct FastAPI app for MLX model serving.

    Args:
        model_name: The name of the model being served.

    Returns:
        The FastAPI application instance.

    """

    def _startup() -> None:
        """Preload model weights during server startup."""
        logger.info("Initializing MLX serve app for model %s", model_name)
        if mx is not None:
            try:
                _load_mlx_model(model_name)
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
                logger.warning("Asynchronous model preload failed for %s: %s", model_name, e)

    return create_common_app(
        backend_name="mlx",
        model_name=model_name,
        startup_callback=_startup,
        generate_logic=lambda prompt: _generate_query(prompt, model_name=model_name),
        batch_generate_logic=lambda prompts: _batch_generate_queries(prompts, model_name=model_name),
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
    if mx is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MLX dependencies are missing for serve.")

    result = serve_model_wrapper(
        backend_name="mlx",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        missing_deps=mx is None,
        missing_status="mocked_missing_mlx",
        app_factory=lambda: _app_factory(model_name),
    )

    if result["status"] == "running_mlx_serve":
        logger.info("Starting MLX server on port %d with max_batch_size %d", port, max_batch_size)

    return result
