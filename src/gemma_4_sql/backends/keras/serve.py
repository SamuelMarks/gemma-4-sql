"""Keras-specific continuous batching inference logic."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_serve import create_backend_serve_app, serve_model_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import FastAPIAppProtocol, JSONDict, JSONValue, KerasCausalLMProtocol, ServeResultDict

logger = logging.getLogger(__name__)


@runtime_checkable
class _GemmaCausalLMLoaderProtocol(Protocol):
    """Protocol for `keras_nlp.models.GemmaCausalLM` exposing `.from_preset(model_name)`."""

    def from_preset(self, model_name: str, /) -> KerasCausalLMProtocol:
        """Load a `GemmaCausalLM` model from a preset identifier."""
        ...


keras = optional_import("keras")
tf = optional_import("tensorflow")
GemmaCausalLM: _GemmaCausalLMLoaderProtocol | None = optional_attr(
    "keras_nlp.models",
    "GemmaCausalLM",
    _GemmaCausalLMLoaderProtocol,
)


def create_app(model_name: str, max_batch_size: int = 32) -> FastAPIAppProtocol:
    """Create the FastAPI application for the Keras server.

    Args:
        model_name: The name of the target model.
        max_batch_size: Maximum batch size per continuous batching window.

    Returns:
        The FastAPI application instance.

    """
    loaded_model: KerasCausalLMProtocol | None = None

    def _startup() -> None:
        """Initialize Keras model on server startup."""
        nonlocal loaded_model
        logger.info("Pre-loading Keras model %s for continuous batching server...", model_name)
        try:
            gemma_causal_lm_cls = GemmaCausalLM if GemmaCausalLM is not None else optional_attr("keras_nlp.models", "GemmaCausalLM", _GemmaCausalLMLoaderProtocol)
            if gemma_causal_lm_cls is None:
                raise ImportError("keras_nlp.models.GemmaCausalLM is not available.")
            loaded_model = gemma_causal_lm_cls.from_preset(model_name)
        except (ImportError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Could not pre-load Keras model %s: %s", model_name, e)

    def _call_keras_generate(model_name: str, prompt: str) -> JSONDict:
        """Generate SQL using pre-loaded Keras model or fallback to `keras.inference.generate_sql`."""
        from gemma_4_sql.backends.keras.inference import generate_sql

        if loaded_model is not None:
            try:
                out = loaded_model.generate(prompt)
                return {"sql": str(out).strip()}
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning("Keras loaded_model.generate failed, falling back to generate_sql: %s", e)

        res = generate_sql(model_name=model_name, prompt=prompt)
        return {k: v for k, v in res.items() if isinstance(v, (str, int, float, bool)) or v is None}

    def _batch_generate(prompts: list[str]) -> list[str]:
        """Generate SQL queries for a batch of prompts using pre-loaded Keras model or per-prompt fallback."""
        from gemma_4_sql.exceptions import InferenceError

        if loaded_model is not None:
            try:
                outputs = loaded_model.generate(prompts)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
                return [str(out).strip() for out in outputs]  # pyright: ignore[reportGeneralTypeIssues]  # ty: ignore[not-iterable]
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning("Keras batch generation failed, falling back to individual generation: %s", e)

        results: list[str] = []
        for p in prompts:
            try:
                out_dict = _call_keras_generate(model_name=model_name, prompt=p)
                sql = out_dict.get("sql", "")
                if not sql:
                    raise InferenceError(f"Keras inference returned empty SQL for prompt '{p}'", backend="keras")
                results.append(str(sql))
            except (InferenceError, RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
                if isinstance(e, InferenceError):
                    raise
                raise InferenceError(f"Keras generation failed: {e}", backend="keras", cause=e) from e
        return results

    return create_backend_serve_app(
        backend_name="keras",
        model_name=model_name,
        generate_sql_fn=_call_keras_generate,
        startup_callback=_startup,
        batch_generate_logic=_batch_generate,
        max_batch_size=max_batch_size,
    )


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, **_kwargs: JSONValue) -> ServeResultDict:
    """Serve a model using Keras continuous batching.

    Args:
        model_name: The name of the target model.
        port: The network port to listen on.
        max_batch_size: The maximum allowed batch size.
        **_kwargs: Underlying server and backend-specific configuration options.

    Returns:
        A structured `ServeResultDict` containing the server metadata.

    Raises:
        DependencyMissingError: If Keras dependencies are missing for serve.

    """
    if tf is None or keras is None:
        raise DependencyMissingError("Keras dependencies are missing for serve.", backend="keras")

    return serve_model_wrapper(
        backend_name="keras",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        app_factory=lambda: create_app(model_name, max_batch_size=max_batch_size),
    )
