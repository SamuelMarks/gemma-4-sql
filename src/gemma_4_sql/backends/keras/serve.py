"""Keras-specific continuous batching inference logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_serve import create_common_app, serve_model_wrapper
from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)

try:
    import keras as _keras
    import tensorflow as _tf

    keras: Any = _keras
    tf: Any = _tf
except (ImportError, AttributeError):
    keras = None
    tf = None
FastAPI = None
Request = None
JSONResponse = None
uvicorn = None
with catch_optional_imports():
    pass


def create_app(model_name: str, *, test_mode: bool = False) -> object:
    """Create the FastAPI application for the Keras server.

    Args:
        model_name: The name of the target model.
        test_mode: Boolean flag indicating test mode.

    Returns:
        The FastAPI application instance.
    """
    loaded_model: Any = None

    def _startup() -> None:
        """Initialize Keras model on server startup."""
        nonlocal loaded_model
        logger.info("Exporting Keras model %s to SavedModel format for TF Serving...", model_name)
        if not test_mode:
            try:
                gemma_causal_lm_cls = __import__("keras_nlp.models", fromlist=["GemmaCausalLM"]).GemmaCausalLM
                loaded_model = gemma_causal_lm_cls.from_preset(model_name)
            except (ImportError, ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("Could not pre-load Keras model %s: %s", model_name, e)

    def _generate(prompt: str) -> str:
        """Generate a SQL query for a single prompt.

        Args:
            prompt: Natural language input query.

        Returns:
            Generated SQL query string.

        Raises:
            InferenceError: If model inference fails during non-test execution.
        """
        if test_mode:
            return f"SELECT * FROM keras_serve WHERE prompt='{prompt}'"
        from gemma_4_sql.backends.keras.inference import generate_sql
        from gemma_4_sql.exceptions import InferenceError

        if loaded_model is not None and hasattr(loaded_model, "generate"):
            try:
                out = loaded_model.generate(prompt)
                return str(out).strip()
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning("Keras loaded_model.generate failed, falling back to generate_sql: %s", e)

        try:
            out = generate_sql(model_name=model_name, prompt=prompt)
            sql = out.get("sql", "")
            if sql:
                return str(sql)
            raise InferenceError(f"Keras inference returned empty SQL for prompt '{prompt}'")
        except Exception as e:
            if isinstance(e, InferenceError):
                raise
            raise InferenceError(f"Keras generation failed: {e}") from e

    def _batch_generate(prompts: list[str]) -> list[str]:
        """Generate SQL queries for a batch of prompts.

        Args:
            prompts: List of natural language input queries.

        Returns:
            List of generated SQL query strings.
        """
        if test_mode:
            return [f"SELECT * FROM keras_serve WHERE prompt='{p}'" for p in prompts]
        if loaded_model is not None and hasattr(loaded_model, "generate"):
            try:
                outputs = loaded_model.generate(prompts)
                return [str(out).strip() for out in outputs]
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning("Keras batch generation failed, falling back to individual generation: %s", e)
        return [_generate(p) for p in prompts]

    return create_common_app(
        backend_name="keras",
        model_name=model_name,
        test_mode=test_mode,
        startup_callback=_startup,
        generate_logic=_generate,
        batch_generate_logic=_batch_generate,
    )


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, **kwargs: JSONValue) -> JSONDict:
    """Serve a model using Keras continuous batching.

    Args:
        model_name: The name of the target model.
        port: The network port to listen on.
        max_batch_size: The maximum allowed batch size.
        **kwargs: Underlying server and backend-specific configuration options.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If Keras dependencies are missing for serve.
    """
    if tf is None or keras is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Keras dependencies are missing for serve.")

    return serve_model_wrapper(
        backend_name="keras",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        missing_deps=False,
        missing_status="mocked_missing_keras",
        app_factory=lambda: create_app(model_name, test_mode=bool(kwargs.get("test_mode"))),
        test_mode=bool(kwargs.get("test_mode")),
    )
