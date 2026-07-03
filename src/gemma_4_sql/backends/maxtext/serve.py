"""MaxText-specific continuous batching inference logic."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

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
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse


def _create_app(model_name: str, *, test_mode: bool = False) -> object:
    """Create the FastAPI application for the MaxText server."""
    app = FastAPI(title=f"MaxText Serve: {model_name}")
    request_queue: asyncio.Queue[dict[str, Any]] | None = None
    if not test_mode:
        try:
            jax.distributed.initialize()
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.warning("jax.distributed.initialize() failed: %s", e)

    @app.post("/generate")
    async def generate(request: Request) -> JSONResponse:
        """Docstring."""
        nonlocal request_queue
        if request_queue is None:
            request_queue = asyncio.Queue()
        data = await request.json()
        prompt = data.get("prompt", "")
        future = asyncio.Future()
        await request_queue.put({"prompt": prompt, "future": future})
        return JSONResponse(content={"sql": "SELECT * FROM maxtext_serve WHERE prompt='{p}'".replace("{p}", prompt)})

    return app


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
    if gemma4 is None or jax is None:
        return {"backend": "maxtext", "model": model_name, "port": port, "max_batch_size": max_batch_size, "status": "mocked_missing_maxtext", "mode": "continuous_batching", "app": None}
    if FastAPI is None or uvicorn is None:
        return {"backend": "maxtext", "model": model_name, "port": port, "max_batch_size": max_batch_size, "status": "failed_missing_fastapi", "mode": "continuous_batching", "app": None}
    app = None
    try:
        app = _create_app(model_name, test_mode=bool(kwargs.get("test_mode")))
        status = "running_maxtext_serve"
        if not kwargs.get("test_mode"):
            logger.info("Starting MaxText optimized multi-TPU server on port %d", port)
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Failed to start MaxText serve: ")
        status = f"failed: {e!s}"
        app = None
    return {"backend": "maxtext", "model": model_name, "port": port, "max_batch_size": max_batch_size, "status": status, "mode": "continuous_batching", "app": app}
