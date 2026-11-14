"""MLX-specific continuous batching inference logic."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_serve import create_backend_serve_app, serve_model_wrapper
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.backends.mlx.inference import MODEL_CACHE as _mlx_model_cache
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError
from gemma_4_sql.type_hints import (
    FastAPIAppProtocol,
    JSONDict,
    JSONValue,
    MlxCausalLMProtocol,
    ServeResultDict,
    TokenizerProtocol,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _MlxLmModuleProtocol(Protocol):
    """Protocol for `mlx_lm` exposing `.load(model_name)`."""

    def load(
        self,
        model_name: str,
        /,
    ) -> tuple[MlxCausalLMProtocol, TokenizerProtocol | None] | MlxCausalLMProtocol:
        """Load an MLX model and optional tokenizer.

        Args:
            model_name: Pretrained model identifier or local path.

        Returns:
            `(model, tokenizer)` pair or `MlxCausalLMProtocol`.

        """
        ...


mx = optional_import("mlx.core")

MAX_MLX_CACHE_SIZE = 4


def clear_mlx_model_cache() -> None:
    """Clear the bounded MLX model and tokenizer cache."""
    _mlx_model_cache.clear()


def _load_mlx_model(model_name: str) -> tuple[MlxCausalLMProtocol, TokenizerProtocol | None]:
    """Load an MLX model and tokenizer into the bounded LRU cache.

    Args:
        model_name: The name or path of the target MLX model.

    Returns:
        A tuple of `(model, tokenizer)`.

    Raises:
        ImportError: If `mlx_lm` is not installed.
        ValueError: If model configuration or weights are invalid.
        OSError: If model files cannot be loaded from disk or hub.
        RuntimeError: If MLX runtime execution fails during model loading.
        AttributeError: If `mlx_lm` lacks expected loader attributes.

    """
    _mlx_model_cache.maxsize = MAX_MLX_CACHE_SIZE
    cached = _mlx_model_cache.get_model(model_name)
    if cached is not None:
        return cached
    try:
        mlx_lm = optional_import("mlx_lm", _MlxLmModuleProtocol)
        if mlx_lm is None:
            raise ImportError("mlx_lm is not installed.")
        loaded = mlx_lm.load(model_name)
        if isinstance(loaded, (tuple, list)):
            model: MlxCausalLMProtocol = loaded[0]
            tokenizer: TokenizerProtocol | None = loaded[1] if len(loaded) > 1 else None
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
        InferenceError: If model inference fails or returns empty SQL.
        RuntimeError: If an underlying runtime error is re-raised during generation.
        ValueError: If invalid generation parameters or values are encountered.
        TypeError: If incompatible argument or return types are encountered.
        KeyError: If expected result dictionary keys are missing.
        AttributeError: If model or tokenizer attributes are missing.
        OSError: If model or tokenizer I/O fails.

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
        logger.exception("MLX serve generation encountered error")
        raise InferenceError(f"MLX generation failed: {e}", backend="mlx", cause=e) from e


def _default_query_generator(prompt: str, model_name: str) -> str:
    """Delegate `(prompt, model_name)` generation to `_generate_query`.

    Args:
        prompt: Natural language SQL prompt.
        model_name: Target model identifier.

    Returns:
        Generated SQL query string.

    """
    return _generate_query(prompt, model_name=model_name)


def _batch_generate_queries(
    prompts: list[str],
    model_name: str = "",
    *,
    query_generator: Callable[[str, str], str] | None = None,
) -> list[str]:
    """Generate SQL queries for a batch of prompts using MLX inference.

    Args:
        prompts: Sequence of natural language prompts.
        model_name: Target model identifier.
        query_generator: Optional injected `(prompt, model_name) -> sql` callable.

    Returns:
        List of generated SQL query strings.

    """
    gen_fn = query_generator if query_generator is not None else _default_query_generator
    return [gen_fn(p, model_name) for p in prompts]


def _app_factory(
    model_name: str,
    max_batch_size: int = 32,
    *,
    model_loader: Callable[[str], tuple[MlxCausalLMProtocol, TokenizerProtocol | None] | None] | None = None,
    query_generator: Callable[[str, str], str] | None = None,
) -> FastAPIAppProtocol:
    """Construct FastAPI app for MLX model serving.

    Args:
        model_name: The name of the model being served.
        max_batch_size: Maximum batch size per continuous batching window.
        model_loader: Optional injected model preload callable for testing or custom runtimes.
        query_generator: Optional injected `(prompt, model_name) -> sql` callable.

    Returns:
        The FastAPI application instance.

    """
    load_fn = model_loader if model_loader is not None else _load_mlx_model
    gen_fn = query_generator if query_generator is not None else _default_query_generator

    def _startup() -> None:
        """Preload model weights during server startup."""
        logger.info("Initializing MLX serve app for model %s", model_name)
        if mx is not None:
            try:
                load_fn(model_name)
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
                logger.warning("Asynchronous model preload failed for %s: %s", model_name, e)

    def _call_mlx_generate(model_name: str, prompt: str) -> JSONDict:
        """Delegate single-prompt generation to `gen_fn`.

        Args:
            model_name: Target model identifier.
            prompt: Natural language SQL prompt.

        Returns:
            JSON dictionary mapping `'sql'` to the generated SQL query string.

        """
        return {"sql": gen_fn(prompt, model_name)}

    return create_backend_serve_app(
        backend_name="mlx",
        model_name=model_name,
        generate_sql_fn=_call_mlx_generate,
        startup_callback=_startup,
        batch_generate_logic=lambda prompts: _batch_generate_queries(
            prompts,
            model_name=model_name,
            query_generator=gen_fn,
        ),
        max_batch_size=max_batch_size,
    )


def serve_mlx_model(
    model_name: str,
    port: int = 8000,
    max_batch_size: int = 256,
    *,
    model_loader: Callable[[str], tuple[MlxCausalLMProtocol, TokenizerProtocol | None] | None] | None = None,
) -> ServeResultDict:
    """Serve a model using MLX continuous batching with optional injected `model_loader`.

    Args:
        model_name: The name of the target model.
        port: The network port to listen on.
        max_batch_size: The maximum allowed batch size.
        model_loader: Optional injected model preload callable for testing or custom runtimes.

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
        app_factory=lambda: _app_factory(
            model_name,
            max_batch_size=max_batch_size,
            model_loader=model_loader,
        ),
    )

    if result["status"] == "running_mlx_serve":
        logger.info("Starting MLX server on port %d with max_batch_size %d", port, max_batch_size)

    return result


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

    """
    return serve_mlx_model(model_name, port=port, max_batch_size=max_batch_size)
