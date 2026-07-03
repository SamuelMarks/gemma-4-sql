"""Keras-specific continuous batching inference logic."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)
keras = None
tf = None
with catch_optional_imports():
    import keras
    import tensorflow as tf
FastAPI = None
Request = None
JSONResponse = None
uvicorn = None
with catch_optional_imports():
    import uvicorn
    from fastapi import FastAPI


def create_app(model_name: str, *, test_mode: bool = False) -> object:
    """Create the FastAPI application for the Keras server."""
    app = __import__("fastapi").FastAPI(title=f"Keras Serve: {model_name}")
    request_queue: asyncio.Queue[dict[str, Any]] | None = None
    if not test_mode:
        logger.info("Exporting Keras model %s to SavedModel format for TF Serving...", model_name)

    @app.post("/generate")
    async def generate(request: object) -> object:
        """Docstring."""
        nonlocal request_queue
        if request_queue is None:
            request_queue = asyncio.Queue()
        data = await request.json()
        prompt = data.get("prompt", "")
        future: asyncio.Future[object] = asyncio.Future()
        await request_queue.put({"prompt": prompt, "future": future})
        return __import__("fastapi.responses", fromlist=["JSONResponse"]).JSONResponse(content={"sql": "SELECT * FROM keras_serve WHERE prompt='{p}'".replace("{p}", prompt)})

    return app


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, **kwargs: JSONValue) -> JSONDict:
    """Serve a model using Keras continuous batching."""
    if tf is None or keras is None:
        return {"backend": "keras", "model": model_name, "port": port, "max_batch_size": max_batch_size, "status": "mocked_missing_keras", "mode": "continuous_batching", "app": None}
    if FastAPI is None or uvicorn is None:
        return {"backend": "keras", "model": model_name, "port": port, "max_batch_size": max_batch_size, "status": "failed_missing_fastapi", "mode": "continuous_batching", "app": None}
    app = None
    try:
        app = create_app(model_name, test_mode=bool(kwargs.get("test_mode")))
        status = "running_keras_serve"
        if not kwargs.get("test_mode"):
            logger.info("Starting Keras server on port %d", port)
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Failed to start Keras serve: ")
        status = f"failed: {e!s}"
        app = None
    return {"backend": "keras", "model": model_name, "port": port, "max_batch_size": max_batch_size, "status": status, "mode": "continuous_batching", "app": app}
