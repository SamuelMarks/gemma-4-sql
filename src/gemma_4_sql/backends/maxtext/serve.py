# Copyright 2024
"""MaxText-specific continuous batching inference logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_serve import create_common_app, serve_model_wrapper
from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)
jax = None
gemma4 = None
with catch_optional_imports():
    import jax
    from maxtext.models import gemma4
FastAPI = None
Request = None
JSONResponse = None
uvicorn = None
with catch_optional_imports():
    pass  # pragma: no cover


def _create_app(model_name: str, *, test_mode: bool = False) -> object:
    """Create the FastAPI application for the MaxText server.

    Returns:
        object: The resulting output from the operation.

    """

    def _startup() -> None:
        try:
            jax.distributed.initialize()
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.warning("jax.distributed.initialize() failed: %s", e)

    return create_common_app(
        backend_name="maxtext",
        model_name=model_name,
        test_mode=test_mode,
        startup_callback=_startup,
    )


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, **kwargs: JSONValue) -> JSONDict:
    """Serve a model using MaxText continuous batching.

    Args:
    ----
        model_name: The name of the model to serve.
        port: The port to bind the server to.
        max_batch_size: The maximum batch size.
        **kwargs: Additional parameters.

    Returns:
    -------
        A dictionary containing serving status and metadata.

    """
    return serve_model_wrapper(
        backend_name="maxtext",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        missing_deps=gemma4 is None or jax is None,
        missing_status="mocked_missing_maxtext",
        app_factory=lambda: _create_app(model_name, test_mode=bool(kwargs.get("test_mode"))),
        test_mode=bool(kwargs.get("test_mode")),
    )
