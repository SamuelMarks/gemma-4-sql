# Copyright 2024
"""Keras-specific continuous batching inference logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_serve import create_common_app, serve_model_wrapper
from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)
keras = None
tf = None
with catch_optional_imports():
    import keras
    import tensorflow as tf  # pragma: no cover
FastAPI = None
Request = None
JSONResponse = None
uvicorn = None
with catch_optional_imports():
    pass  # pragma: no cover


def create_app(model_name: str, *, test_mode: bool = False) -> object:
    """Create the FastAPI application for the Keras server.

    Returns:
        object: The resulting output from the operation.

    """

    def _startup() -> None:
        """Execute function."""
        logger.info("Exporting Keras model %s to SavedModel format for TF Serving...", model_name)

    return create_common_app(
        backend_name="keras",
        model_name=model_name,
        test_mode=test_mode,
        startup_callback=_startup,
    )


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, **kwargs: JSONValue) -> JSONDict:
    """Serve a model using Keras continuous batching.

    Returns:
        object: The resulting output from the operation.

    """
    return serve_model_wrapper(
        backend_name="keras",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        missing_deps=tf is None or keras is None,
        missing_status="mocked_missing_keras",
        app_factory=lambda: create_app(model_name, test_mode=bool(kwargs.get("test_mode"))),
        test_mode=bool(kwargs.get("test_mode")),
    )
