"""Common FastAPI serving utilities for backends."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError
from gemma_4_sql.type_hints import JSONDict

if TYPE_CHECKING:
    from fastapi import Request

    from gemma_4_sql.type_hints import ServeResultDict
else:
    Request = optional_attr("fastapi", "Request")

logger = logging.getLogger(__name__)

uvicorn = optional_import("uvicorn")
FastAPI = optional_attr("fastapi", "FastAPI")
HTTPException = optional_attr("fastapi", "HTTPException")
JSONResponse = optional_attr("fastapi.responses", "JSONResponse")


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

    if startup_callback is not None:
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
                if not item["future"].done():
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

        req_any: Any = request
        data = await req_any.json()
        validated_req = GenerateRequest.from_dict(cast(dict[str, Any], data))
        prompt = validated_req.prompt
        if validated_req.image_base64 or validated_req.audio_base64 or validated_req.image_path or validated_req.audio_path:
            from gemma_4_sql.backends.common_multimodal import format_multimodal_prompt

            formatted = format_multimodal_prompt(
                prompt,
                has_image=bool(validated_req.image_base64 or validated_req.image_path),
                has_audio=bool(validated_req.audio_base64 or validated_req.audio_path),
            )
            prompt = str(formatted.get("prompt", prompt))

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
        content: dict[str, Any] = {
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


def create_backend_serve_app(
    backend_name: str,
    model_name: str,
    generate_sql_fn: Callable[..., JSONDict],
    startup_callback: Callable[[], None] | None = None,
    batch_generate_logic: Callable[[list[str]], list[str]] | None = None,
    max_batch_size: int = 32,
) -> object:
    """Create a FastAPI continuous batching application wrapping a backend's `generate_sql` function.

    Args:
        backend_name: Backend identifier ('jax', 'keras', 'maxtext', 'mlx', 'pytorch').
        model_name: Target model name.
        generate_sql_fn: Backend `generate_sql(model_name=..., prompt=...)` callable.
        startup_callback: Optional callback invoked during app initialization.
        batch_generate_logic: Optional batch generation override callable.
        max_batch_size: Maximum batch size per dispatch window.

    Returns:
        Configured FastAPI application instance.

    """

    def _generate(prompt: str) -> str:
        """Execute backend SQL generation and validate non-empty output."""
        try:
            out = generate_sql_fn(model_name=model_name, prompt=prompt)
            sql = out.get("sql", "")
            if sql:
                return str(sql)
            msg = f"{backend_name.title()} inference returned empty SQL for prompt '{prompt}'"
            raise InferenceError(msg, backend=backend_name)
        except Exception as e:
            if isinstance(e, InferenceError):
                raise
            msg = f"{backend_name.title()} generation failed: {e}"
            raise InferenceError(msg, backend=backend_name, cause=e) from e

    return create_common_app(
        backend_name=backend_name,
        model_name=model_name,
        startup_callback=startup_callback,
        generate_logic=_generate,
        batch_generate_logic=batch_generate_logic,
        max_batch_size=max_batch_size,
    )


def serve_model_wrapper(
    backend_name: str,
    model_name: str,
    port: int,
    max_batch_size: int,
    missing_deps: bool,
    app_factory: Callable[[], object],
    run_server: bool = False,
    host: str = "0.0.0.0",
) -> ServeResultDict:
    """Wrap serving logic to unify exception handling and result formatting.

    Args:
        backend_name: The name of the backend engine.
        model_name: Target model identifier.
        port: Listening network port.
        max_batch_size: Maximum concurrency/batch size.
        missing_deps: Whether required dependencies are missing.
        app_factory: Factory callback to create the FastAPI application.
        run_server: Whether to run the uvicorn HTTP server.
        host: Network host binding interface.

    Returns:
        The execution status dictionary.

    Raises:
        DependencyMissingError: If FastAPI, uvicorn, or backend dependencies are missing.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    if missing_deps:
        raise DependencyMissingError(f"Dependencies missing for backend: {backend_name}")

    if FastAPI is None or uvicorn is None:
        msg = "FastAPI and uvicorn are required for serving."
        raise DependencyMissingError(msg)

    app = None
    try:
        app = app_factory()
        status = f"running_{backend_name}_serve"
        logger.info("Starting %s server on port %d", backend_name.title(), port)
        if run_server:
            uvicorn.run(app, host=host, port=port)
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
