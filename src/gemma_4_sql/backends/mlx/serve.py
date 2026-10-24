"""MLX-specific continuous batching inference logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_inference import BoundedLRUCache
from gemma_4_sql.backends.common_serve import create_backend_serve_app, serve_model_wrapper
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError
from gemma_4_sql.type_hints import MlxCausalLMProtocol

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import FastAPIAppProtocol, JSONDict, JSONValue, ServeResultDict

logger = logging.getLogger(__name__)

mx = optional_import("mlx.core")

MAX_MLX_CACHE_SIZE = 4
_mlx_model_cache: BoundedLRUCache[tuple[Any, Any]] = BoundedLRUCache(maxsize=MAX_MLX_CACHE_SIZE)


def clear_mlx_model_cache() -> None:
    """Clear the bounded MLX model and tokenizer cache."""
    _mlx_model_cache.clear()


def _load_mlx_model(model_name: str) -> tuple[Any, Any]:
    """Load an MLX model and tokenizer into the bounded LRU cache.

    Args:
        model_name: The name or path of the target MLX model.

    Returns:
        A tuple of `(model, tokenizer)`.

    Raises:
        Exception: If `mlx_lm.load` fails.

    """
    _mlx_model_cache.maxsize = MAX_MLX_CACHE_SIZE
    cached = _mlx_model_cache.get_model(model_name)
    if cached is not None:
        return cached
    try:
        import mlx_lm

        loaded: Any = mlx_lm.load(model_name)
        if isinstance(loaded, (tuple, list)):
            model: MlxCausalLMProtocol = loaded[0]
            tokenizer = loaded[1] if len(loaded) > 1 else None
        else:
            model = loaded
            tokenizer = None
        _mlx_model_cache.put_model(model_name, (model, tokenizer))
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
        InferenceError: If model inference fails during execution.

    """
    import gemma_4_sql.backends.mlx.inference as mlx_inf

    try:
        res = mlx_inf.generate_sql(model_name=model_name or "default", prompt=prompt)
        sql = res.get("sql", "")
        if sql:
            return str(sql)
        raise InferenceError(f"MLX inference returned empty SQL for prompt '{prompt}'", backend="mlx")
    except (InferenceError, RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        if isinstance(e, InferenceError):
            raise
        logger.error("MLX serve generation encountered error: %s", e)
        raise InferenceError(f"MLX generation failed: {e}", backend="mlx", cause=e) from e


def _batch_generate_queries(prompts: list[str], model_name: str = "") -> list[str]:
    """Generate SQL queries for a batch of prompts using MLX inference.

    Args:
        prompts: Sequence of natural language prompts.
        model_name: Target model identifier.

    Returns:
        List of generated SQL query strings.

    """
    return [_generate_query(p, model_name=model_name) for p in prompts]


def _app_factory(model_name: str, max_batch_size: int = 32) -> FastAPIAppProtocol:
    """Construct FastAPI app for MLX model serving.

    Args:
        model_name: The name of the model being served.
        max_batch_size: Maximum batch size per continuous batching window.

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

    def _call_mlx_generate(model_name: str, prompt: str) -> JSONDict:
        """Delegate single-prompt generation to `_generate_query`."""
        return {"sql": _generate_query(prompt, model_name=model_name)}

    return create_backend_serve_app(
        backend_name="mlx",
        model_name=model_name,
        generate_sql_fn=_call_mlx_generate,
        startup_callback=_startup,
        batch_generate_logic=lambda prompts: _batch_generate_queries(prompts, model_name=model_name),
        max_batch_size=max_batch_size,
    )


def serve_model(
    model_name: str,
    port: int = 8000,
    max_batch_size: int = 256,
    **_kwargs: JSONValue,
) -> ServeResultDict:
    """Serve a model using MLX continuous batching.

    Args:
        model_name: The name of the target model.
        port: The network port to listen on.
        max_batch_size: The maximum allowed batch size.
        **_kwargs: Underlying server and backend-specific configuration options.

    Returns:
        A structured `ServeResultDict` containing the server metadata.

    Raises:
        DependencyMissingError: If MLX dependencies are not installed.

    """
    if mx is None:
        raise DependencyMissingError("MLX dependencies are missing for serve.", backend="mlx")

    result = serve_model_wrapper(
        backend_name="mlx",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        missing_deps=mx is None,
        app_factory=lambda: _app_factory(model_name, max_batch_size=max_batch_size),
    )

    if result["status"] == "running_mlx_serve":
        logger.info("Starting MLX server on port %d with max_batch_size %d", port, max_batch_size)

    return result
