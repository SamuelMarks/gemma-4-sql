"""Common FastAPI serving utilities for backends."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import JSONDict

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

try:
    import uvicorn as _uvicorn
    from fastapi import FastAPI as _FastAPI
    from fastapi import HTTPException as _HTTPException
    from fastapi import Request as _Request
    from fastapi.responses import JSONResponse as _JSONResponse

    uvicorn: Any = _uvicorn
    FastAPI: Any = _FastAPI
    HTTPException: Any = _HTTPException
    Request: Any = _Request
    JSONResponse: Any = _JSONResponse
except (ImportError, AttributeError):
    uvicorn = None
    FastAPI = None
    HTTPException = None
    Request = None
    JSONResponse = None


@dataclass
class GenerateRequest:
    """Schema for SQL generation requests.

    Attributes:
        prompt: Natural language question to translate into SQL.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        image_base64: Base64-encoded image bytes for visual schema/diagram context.
        audio_base64: Base64-encoded audio bytes for spoken natural language queries.
        image_path: Optional filesystem path to schema image.
        audio_path: Optional filesystem path to audio recording.
        modality: Explicit modality selector ('text', 'vision', 'audio', 'multimodal').
    """

    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    image_base64: str | None = None
    audio_base64: str | None = None
    image_path: str | None = None
    audio_path: str | None = None
    modality: str = "text"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerateRequest:
        """Construct and validate GenerateRequest from payload dictionary.

        Args:
            data: Raw input dictionary from JSON request body.

        Returns:
            Validated GenerateRequest instance.

        Raises:
            ValueError: If prompt is missing or not a string.
        """
        prompt = data.get("prompt")
        if prompt is None or not isinstance(prompt, str):
            msg = "Field 'prompt' must be a valid string."
            raise ValueError(msg)
        max_tokens = int(data.get("max_tokens", 128))
        temperature = float(data.get("temperature", 0.0))
        image_base64 = data.get("image_base64") or data.get("image")
        audio_base64 = data.get("audio_base64") or data.get("audio")
        image_path = data.get("image_path")
        audio_path = data.get("audio_path")
        modality = str(data.get("modality", "text"))
        return cls(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            image_base64=str(image_base64) if image_base64 else None,
            audio_base64=str(audio_base64) if audio_base64 else None,
            image_path=str(image_path) if image_path else None,
            audio_path=str(audio_path) if audio_path else None,
            modality=modality,
        )


def create_common_app(
    backend_name: str,
    model_name: str,
    test_mode: bool = False,
    startup_callback: Callable[[], None] | None = None,
    generate_logic: Callable[[str], str] | None = None,
    batch_generate_logic: Callable[[list[str]], list[str]] | None = None,
    max_batch_size: int = 32,
    max_wait_ms: float = 10.0,
    require_handlers: bool = False,
) -> object:
    """Create a common FastAPI application for model serving with continuous batching.

    Args:
        backend_name: The name of the backend (e.g., 'keras', 'maxtext', 'jax', 'pytorch').
        model_name: The name of the model being served.
        test_mode: Whether running in test mode.
        startup_callback: Optional logic to run during initialization.
        generate_logic: Optional single-prompt generation callable.
        batch_generate_logic: Optional batch generation callable accepting list of prompts.
        max_batch_size: The maximum number of requests bundled per batch.
        max_wait_ms: Maximum wait duration in milliseconds before dispatching a partial batch.
        require_handlers: Whether to require at least one generation handler at app construction.

    Returns:
        A FastAPI application instance.

    Raises:
        ValueError: If require_handlers is True and no generation handler is supplied.
    """
    if require_handlers and generate_logic is None and batch_generate_logic is None:
        msg = f"At least one generation logic callback must be provided for backend '{backend_name}'."
        raise ValueError(msg)

    app = FastAPI(title=f"{backend_name.title()} Serve: {model_name}")
    request_queue: asyncio.Queue[dict[str, Any]] | None = None
    worker_task: asyncio.Task[None] | None = None

    if not test_mode and startup_callback is not None:
        startup_callback()

    async def _batching_worker() -> None:
        """Background worker that bundles incoming requests and invokes batched inference."""
        nonlocal request_queue
        assert request_queue is not None
        while True:
            first_item = await request_queue.get()
            batch = [first_item]

            end_time = asyncio.get_event_loop().time() + (max_wait_ms / 1000.0)
            while len(batch) < max_batch_size:
                timeout = max(0.0, end_time - asyncio.get_event_loop().time())
                try:
                    next_item = await asyncio.wait_for(request_queue.get(), timeout=timeout)
                    batch.append(next_item)
                except (TimeoutError, asyncio.TimeoutError):
                    break

            prompts = [item["prompt"] for item in batch]
            try:
                if batch_generate_logic is not None:
                    sql_responses = batch_generate_logic(prompts)
                    if len(sql_responses) != len(prompts):
                        msg = f"Batch generation returned {len(sql_responses)} results for {len(prompts)} prompts."
                        raise ValueError(msg)
                elif generate_logic is not None:
                    sql_responses = [generate_logic(p) for p in prompts]
                else:
                    msg = f"No generation logic registered for backend '{backend_name}'."
                    raise NotImplementedError(msg)
            except (ValueError, TypeError, RuntimeError, OSError, AttributeError, KeyError, DependencyMissingError, NotImplementedError) as e:
                for item in batch:
                    if not item["future"].done():
                        item["future"].set_exception(e)
                continue

            for item, sql in zip(batch, sql_responses):
                item["future"].set_result(sql)

    @app.post("/generate", response_model=None)
    async def generate(request: Request) -> Any:
        """Handle continuous batching Text-to-SQL generation.

        Args:
            request: The incoming HTTP request.

        Returns:
            A JSON response containing the generated SQL.

        Raises:
            CancelledError: If generation is cancelled.
            NotImplementedError: If no generation logic is available.
            ValueError: If request payload is invalid.
        """
        nonlocal request_queue, worker_task
        if request_queue is None:
            request_queue = asyncio.Queue()
        if worker_task is None or worker_task.done():
            try:
                loop = asyncio.get_running_loop()
                worker_task = loop.create_task(_batching_worker())
            except RuntimeError:
                worker_task = None

        data = await request.json()
        validated_req = GenerateRequest.from_dict(data)
        prompt = validated_req.prompt
        if validated_req.image_base64 or validated_req.audio_base64 or validated_req.image_path or validated_req.audio_path:
            from gemma_4_sql.backends.common_multimodal import format_multimodal_prompt

            formatted = format_multimodal_prompt(
                prompt,
                has_image=bool(validated_req.image_base64 or validated_req.image_path),
                has_audio=bool(validated_req.audio_base64 or validated_req.audio_path),
            )
            prompt = formatted["prompt"]

        future: asyncio.Future[Any] = asyncio.Future()
        await request_queue.put({"prompt": prompt, "future": future})

        if worker_task is not None and not worker_task.done():
            try:
                sql_response = await future
            except asyncio.CancelledError:
                future.cancel()
                raise
        elif generate_logic is not None:
            sql_response = generate_logic(prompt)
        elif batch_generate_logic is not None:
            sql_responses = batch_generate_logic([prompt])
            sql_response = sql_responses[0]
        else:
            msg = f"No generation logic registered for backend '{backend_name}'."
            raise NotImplementedError(msg)

        res_payload = {"sql": sql_response, "modality": validated_req.modality}
        if JSONResponse is not None:
            return JSONResponse(content=res_payload)
        return res_payload

    @app.get("/health", response_model=None)
    async def health() -> Any:
        """Return server health status.

        Returns:
            A JSON response with health information.
        """
        queue_depth = request_queue.qsize() if request_queue is not None else 0
        content = {
            "status": "healthy",
            "backend": backend_name,
            "model": model_name,
            "queue_depth": queue_depth,
        }
        if JSONResponse is not None:
            return JSONResponse(content=content)
        return content

    @app.get("/ready", response_model=None)
    async def ready() -> Any:
        """Return server readiness and queue depth status.

        Returns:
            A JSON response with readiness metrics.
        """
        queue_depth = request_queue.qsize() if request_queue is not None else 0
        content = {
            "status": "ready",
            "backend": backend_name,
            "model": model_name,
            "queue_depth": queue_depth,
            "max_batch_size": max_batch_size,
        }
        if JSONResponse is not None:
            return JSONResponse(content=content)
        return content

    @app.get("/v1/models", response_model=None)
    async def list_models() -> Any:
        """Return available models in OpenAI format.

        Returns:
            A JSON response listing served models.
        """
        content = {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "owned_by": backend_name,
                    "permission": [],
                }
            ],
        }
        if JSONResponse is not None:
            return JSONResponse(content=content)
        return content

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
    run_server: bool = False,
    host: str = "0.0.0.0",
) -> JSONDict:
    """Wrap serving logic to unify exception handling and result formatting.

    Args:
        backend_name: The name of the backend engine.
        model_name: Target model identifier.
        port: Listening network port.
        max_batch_size: Maximum concurrency/batch size.
        missing_deps: Whether required dependencies are missing.
        missing_status: Status message to return when dependencies are missing.
        app_factory: Factory callback to create the FastAPI application.
        test_mode: Whether to run in test mode without starting the network server.
        run_server: Whether to run the uvicorn HTTP server.
        host: Network host binding interface.

    Returns:
        The execution status dictionary.

    Raises:
        DependencyMissingError: If FastAPI or uvicorn is missing.
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

        msg = "FastAPI and uvicorn are required for serving."
        raise DependencyMissingError(msg)

    app = None
    try:
        app = app_factory()
        status = f"running_{backend_name}_serve"
        if not test_mode:
            logger.info("Starting %s server on port %d", backend_name.title(), port)
            if run_server:
                uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.exception("Failed to start %s serve: ", backend_name)
        status = f"failed: {e!s}"
        app = None

    return cast(
        JSONDict,
        {
            "backend": backend_name,
            "model": model_name,
            "port": port,
            "max_batch_size": max_batch_size,
            "status": status,
            "mode": "continuous_batching",
            "app": app,
        },
    )
