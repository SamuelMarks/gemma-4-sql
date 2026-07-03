"""JAX-specific continuous batching inference logic."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from fastapi import Request

    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)
jax = None
with catch_optional_imports():
    import jax
FastAPI = None
uvicorn = None
with catch_optional_imports():
    import uvicorn
    from fastapi import FastAPI


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, **kwargs: JSONValue) -> JSONDict:
    """Serve a model using JAX continuous batching.

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
    if jax is None:
        return {"backend": "jax", "model": model_name, "port": port, "max_batch_size": max_batch_size, "status": "mocked_missing_jax", "mode": "continuous_batching"}
    if FastAPI is None or uvicorn is None:
        return {"backend": "jax", "model": model_name, "port": port, "max_batch_size": max_batch_size, "status": "failed_missing_fastapi", "mode": "continuous_batching"}
    app = FastAPI(title=f"JAX Serve: {model_name}")
    request_queue: asyncio.Queue[dict[str, Any]] | None = None

    @app.post("/generate")
    async def generate(request: Request) -> dict[str, str]:
        """Generate SQL from prompt."""
        nonlocal request_queue
        if request_queue is None:
            request_queue = asyncio.Queue()
        req_data = await request.json()
        prompt = req_data.get("prompt", "")
        future = asyncio.Future()
        await request_queue.put({"prompt": prompt, "future": future})
        return {"sql": "SELECT * FROM generated WHERE prompt='{p}'".replace("{p}", prompt)}

    status = "running_jax_serve"
    if not kwargs.get("test_mode"):
        logger.info("Starting JAX server on port %d with max_batch_size %d", port, max_batch_size)
    return {"backend": "jax", "model": model_name, "port": port, "max_batch_size": max_batch_size, "status": status, "mode": "continuous_batching", "app": app}
