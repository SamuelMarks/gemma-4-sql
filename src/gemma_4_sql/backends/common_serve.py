"""Common FastAPI serving utilities for backends."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemma_4_sql.type_hints import JSONDict

logger = logging.getLogger(__name__)

FastAPI = None
JSONResponse = None
uvicorn = None

with catch_optional_imports():
    import uvicorn
    from fastapi import FastAPI, Request  # pragma: no cover
    from fastapi.responses import JSONResponse  # pragma: no cover


def create_common_app(
    backend_name: str,
    model_name: str,
    test_mode: bool = False,
    startup_callback: Callable[[], None] | None = None,
    generate_logic: Callable[[str], str] | None = None,
) -> object:
    """Create a common FastAPI application for model serving.

    Args:
    ----
        backend_name: The name of the backend (e.g., 'keras', 'maxtext', 'jax', 'pytorch').
        model_name: The name of the model being served.
        test_mode: Whether running in test mode.
        startup_callback: Optional logic to run during initialization.
        generate_logic: Optional logic to execute inside the /generate route.

    Returns:
    -------
        A FastAPI application instance.

    """
    app = FastAPI(title=f"{backend_name.title()} Serve: {model_name}")
    request_queue: asyncio.Queue[dict[str, Any]] | None = None

    if not test_mode and startup_callback is not None:
        startup_callback()

    @app.post("/generate")
    async def generate(request: Request) -> JSONResponse:
        """Execute function.

        Returns:
            The execution result.

        """
        nonlocal request_queue
        if request_queue is None:  # pragma: no cover
            request_queue = asyncio.Queue()
        data = await request.json()
        prompt = data.get("prompt", "")
        future: asyncio.Future[Any] = asyncio.Future()
        await request_queue.put({"prompt": prompt, "future": future})

        sql_response = generate_logic(prompt) if generate_logic else f"SELECT * FROM {backend_name}_serve WHERE prompt='{prompt}'"
        return JSONResponse(content={"sql": sql_response})

    return app


def serve_model_wrapper(
    backend_name: str,
    model_name: str,
    port: int,
    max_batch_size: int,
    missing_deps: bool,
    missing_status: str,
    app_factory: Callable[[], object],
    test_mode: bool = False,
) -> JSONDict:
    """Wrap serving logic to unify exception handling and result formatting.

    Returns:
        The execution result.

    """
    if missing_deps:
        return {
            "backend": backend_name,
            "model": model_name,
            "port": port,
            "max_batch_size": max_batch_size,
            "status": missing_status,
            "mode": "continuous_batching",
            "app": None,
        }

    if FastAPI is None or uvicorn is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("FastAPI and uvicorn are required for serving.")

    app = None
    try:
        app = app_factory()
        status = f"running_{backend_name}_serve"
        if not test_mode:
            logger.info("Starting %s server on port %d", backend_name.title(), port)
    except Exception as e:
        logger.exception("Failed to start %s serve: ", backend_name)
        status = f"failed: {e!s}"
        app = None

    return {
        "backend": backend_name,
        "model": model_name,
        "port": port,
        "max_batch_size": max_batch_size,
        "status": status,
        "mode": "continuous_batching",
        "app": app,
    }
