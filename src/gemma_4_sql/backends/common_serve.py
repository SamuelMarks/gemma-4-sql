"""Common FastAPI serving utilities for backends."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from starlette.requests import Request

from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import (
    OPERATIONAL_EXCEPTIONS,
    ConfigurationError,
    DependencyMissingError,
    ExecutionStatus,
    InferenceError,
    ServingError,
    StatusValue,
    resolve_raise_on_error,
)
from gemma_4_sql.type_hints import (
    FastAPIAppProtocol,
    JSONDict,
    JSONValue,
    ServeResultDict,
    extract_float_kwarg,
    extract_int_kwarg,
    extract_str_kwarg,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class DirectRunnableAppProtocol(FastAPIAppProtocol, Protocol):
    """Protocol for a FastAPI app augmented with a synchronous `.run_direct(prompt)` helper."""

    run_direct: Callable[[str], str]


@runtime_checkable
class FastAPIConstructorProtocol(Protocol):
    """Protocol for the `FastAPI` class constructor."""

    def __call__(self, *, title: str, lifespan: Callable[..., contextlib.AbstractAsyncContextManager[None]]) -> DirectRunnableAppProtocol:
        """Instantiate a FastAPI application."""
        ...


@runtime_checkable
class JSONResponseConstructorProtocol(Protocol):
    """Protocol for the `JSONResponse` class constructor."""

    def __call__(self, *, content: dict[str, JSONValue]) -> JSONValue:
        """Instantiate a JSONResponse."""
        ...


@runtime_checkable
class UvicornModuleProtocol(Protocol):
    """Protocol for the `uvicorn` server module exposing `.run(...)`."""

    def run(self, app: FastAPIAppProtocol, *, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Run an ASGI application with uvicorn."""
        ...


uvicorn = optional_import("uvicorn")
FastAPI: FastAPIConstructorProtocol | None = optional_attr("fastapi", "FastAPI", FastAPIConstructorProtocol)
HTTPException = optional_attr("fastapi", "HTTPException")
JSONResponse: JSONResponseConstructorProtocol | None = optional_attr("fastapi.responses", "JSONResponse", JSONResponseConstructorProtocol)


@dataclass
class _QueuedBatchItem:
    """Strongly-typed queue item for continuous batching."""

    prompt: str
    kwargs: dict[str, JSONValue]
    future: asyncio.Future[str]


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
    def from_dict(cls, data: Mapping[str, JSONValue]) -> GenerateRequest:
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
        max_tokens = extract_int_kwarg(data, "max_tokens", 128)
        temperature = extract_float_kwarg(data, "temperature", 0.0)
        image_base64 = data.get("image_base64") or data.get("image")
        audio_base64 = data.get("audio_base64") or data.get("audio")
        image_path = extract_str_kwarg(data, "image_path")
        audio_path = extract_str_kwarg(data, "audio_path")
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
    generate_logic: Callable[..., str] | None = None,
    batch_generate_logic: Callable[[list[str]], list[str]] | None = None,
    max_batch_size: int = 32,
    max_wait_ms: float = 10.0,
    require_handlers: bool = False,
    request_timeout_s: float = 30.0,
) -> DirectRunnableAppProtocol:
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
        request_timeout_s: Maximum timeout in seconds for an individual batched generation request.

    Returns:
        A FastAPI application instance.

    Raises:
        ValueError: If require_handlers is True and no generation handler is supplied.

    """
    if require_handlers and generate_logic is None and batch_generate_logic is None:
        msg = f"At least one generation logic callback must be provided for backend '{backend_name}'."
        raise ValueError(msg)

    request_queue: asyncio.Queue[_QueuedBatchItem] | None = None
    worker_task: asyncio.Task[None] | None = None

    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPIAppProtocol) -> AsyncGenerator[None, None]:
        """Manage continuous batching background worker lifecycle on FastAPI startup and shutdown."""
        nonlocal request_queue, worker_task
        try:
            yield
        finally:
            task = worker_task
            worker_task = None
            request_queue = None
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    assert callable(FastAPI)
    app: DirectRunnableAppProtocol = FastAPI(title=f"{backend_name.title()} Serve: {model_name}", lifespan=_lifespan)

    if startup_callback is not None:
        startup_callback()

    def _invoke_single(fn: Callable[..., str], prompt_text: str, req_kwargs: Mapping[str, JSONValue] | None = None) -> str:
        """Invoke single-prompt generation callback, forwarding `req_kwargs` if supported."""
        if req_kwargs:
            try:
                return fn(prompt_text, **req_kwargs)
            except TypeError:
                return fn(prompt_text)
        return fn(prompt_text)

    def _run_seq(fn: Callable[..., str], items: list[_QueuedBatchItem]) -> list[str]:
        """Run single-prompt generation sequentially across a batch of request items."""
        return [_invoke_single(fn, it.prompt, it.kwargs) for it in items]

    def _run_direct(prompt_text: str) -> str:
        """Execute synchronous single-prompt generation directly without queueing."""
        if generate_logic is not None:
            return generate_logic(prompt_text)
        if batch_generate_logic is not None:
            sql_responses = batch_generate_logic([prompt_text])
            return sql_responses[0]
        msg = f"No generation logic registered for backend '{backend_name}'."
        raise NotImplementedError(msg)

    async def _batching_worker() -> None:
        """Background worker that bundles incoming requests and invokes batched inference."""
        nonlocal request_queue
        assert request_queue is not None
        while True:
            first_item = await request_queue.get()
            batch = [first_item]

            end_time = asyncio.get_running_loop().time() + (max_wait_ms / 1000.0)
            while len(batch) < max_batch_size:
                timeout = max(0.0, end_time - asyncio.get_running_loop().time())
                try:
                    next_item = await asyncio.wait_for(request_queue.get(), timeout=timeout)
                    batch.append(next_item)
                except TimeoutError:
                    break

            prompts = [item.prompt for item in batch]
            has_custom_kwargs = any(bool(item.kwargs) for item in batch)
            try:
                if has_custom_kwargs and generate_logic is not None:
                    sql_responses = await asyncio.to_thread(_run_seq, generate_logic, batch)
                elif batch_generate_logic is not None:
                    sql_responses = await asyncio.to_thread(batch_generate_logic, prompts)
                    if len(sql_responses) != len(prompts):
                        msg = f"Batch generation returned {len(sql_responses)} results for {len(prompts)} prompts."
                        raise ValueError(msg)
                elif generate_logic is not None:
                    sql_responses = await asyncio.to_thread(_run_seq, generate_logic, batch)
                else:
                    msg = f"No generation logic registered for backend '{backend_name}'."
                    raise NotImplementedError(msg)
            except (ValueError, TypeError, RuntimeError, OSError, AttributeError, KeyError, DependencyMissingError, NotImplementedError) as e:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)
                continue

            for item, sql in zip(batch, sql_responses):
                if not item.future.done():
                    item.future.set_result(sql)

    @app.post("/generate", response_model=None)
    async def generate(request: Request) -> JSONValue:
        """Handle continuous batching Text-to-SQL generation.

        Args:
            request: The incoming HTTP request.

        Returns:
            A JSON response containing the generated SQL.

        Raises:
            CancelledError: If generation is cancelled.
            NotImplementedError: If no generation logic is available.
            TimeoutError: If batched generation exceeds `request_timeout_s`.
            ValueError: If request payload is invalid.

        """
        nonlocal request_queue, worker_task
        loop = asyncio.get_running_loop()

        if request_queue is None or (worker_task is not None and worker_task.get_loop() is not loop):
            request_queue = asyncio.Queue()
            worker_task = None
        if worker_task is None or worker_task.done():
            worker_task = loop.create_task(_batching_worker())

        data: Mapping[str, JSONValue] = await request.json()
        validated_req = GenerateRequest.from_dict(data)
        prompt = validated_req.prompt
        req_kwargs: dict[str, JSONValue] = {}
        if validated_req.image_base64 or validated_req.audio_base64 or validated_req.image_path or validated_req.audio_path:
            from gemma_4_sql.backends.common_multimodal import format_multimodal_prompt

            formatted = format_multimodal_prompt(
                prompt,
                has_image=bool(validated_req.image_base64 or validated_req.image_path),
                has_audio=bool(validated_req.audio_base64 or validated_req.audio_path),
            )
            prompt = str(formatted.get("prompt", prompt))
            req_kwargs.update({
                k: v
                for k, v in (
                    ("image_bytes", validated_req.image_base64),
                    ("audio_clip", validated_req.audio_base64),
                    ("image_path", validated_req.image_path),
                    ("audio_path", validated_req.audio_path),
                )
                if v
            })
        if validated_req.modality != "text":
            req_kwargs["modality"] = validated_req.modality
        if validated_req.temperature != 0.0:
            req_kwargs["temperature"] = validated_req.temperature
        if validated_req.max_tokens != 128:
            req_kwargs["max_length"] = validated_req.max_tokens

        future: asyncio.Future[str] = loop.create_future()
        await request_queue.put(_QueuedBatchItem(prompt=prompt, kwargs=req_kwargs, future=future))

        try:
            sql_response = await asyncio.wait_for(future, timeout=request_timeout_s)
        except asyncio.CancelledError:
            future.cancel()
            raise

        res_payload: dict[str, JSONValue] = {"sql": sql_response, "modality": validated_req.modality}
        if callable(JSONResponse):
            return JSONResponse(content=res_payload)
        return res_payload

    app.run_direct = _run_direct

    @app.get("/health", response_model=None)
    async def health() -> JSONValue:
        """Return server health status.

        Returns:
            A JSON response with health information.

        """
        queue_depth = request_queue.qsize() if request_queue is not None else 0
        content: dict[str, JSONValue] = {
            "status": "healthy",
            "backend": backend_name,
            "model": model_name,
            "queue_depth": queue_depth,
        }
        if callable(JSONResponse):
            return JSONResponse(content=content)
        return content

    @app.get("/ready", response_model=None)
    async def ready() -> JSONValue:
        """Return server readiness and queue depth status.

        Returns:
            A JSON response with readiness metrics.

        """
        queue_depth = request_queue.qsize() if request_queue is not None else 0
        content: dict[str, JSONValue] = {
            "status": "ready",
            "backend": backend_name,
            "model": model_name,
            "queue_depth": queue_depth,
            "max_batch_size": max_batch_size,
        }
        if callable(JSONResponse):
            return JSONResponse(content=content)
        return content

    @app.get("/v1/models", response_model=None)
    async def list_models() -> JSONValue:
        """Return available models in OpenAI format.

        Returns:
            A JSON response listing served models.

        """
        content: dict[str, JSONValue] = {
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
        if callable(JSONResponse):
            return JSONResponse(content=content)
        return content

    return app


def create_backend_serve_app[ResT: Mapping[str, JSONValue]](
    backend_name: str,
    model_name: str,
    generate_sql_fn: Callable[..., ResT],
    startup_callback: Callable[[], None] | None = None,
    batch_generate_logic: Callable[[list[str]], list[str]] | None = None,
    max_batch_size: int = 32,
) -> FastAPIAppProtocol:
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

    def _generate(prompt: str, **gen_kwargs: JSONValue) -> str:
        """Execute backend SQL generation and validate non-empty output."""
        display_name = {"jax": "JAX", "mlx": "MLX", "maxtext": "MaxText", "pytorch": "PyTorch", "keras": "Keras"}.get(
            backend_name.lower(),
            backend_name.title(),
        )
        try:
            if gen_kwargs:
                try:
                    out = generate_sql_fn(model_name=model_name, prompt=prompt, **gen_kwargs)
                except TypeError:
                    out = generate_sql_fn(model_name=model_name, prompt=prompt)
            else:
                out = generate_sql_fn(model_name=model_name, prompt=prompt)
            sql = out.get("sql", "")
            if sql:
                return str(sql)
            msg = f"{display_name} inference returned empty SQL for prompt '{prompt}'"
            raise InferenceError(msg, backend=backend_name)
        except (InferenceError, RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            if isinstance(e, InferenceError):
                raise
            msg = f"{display_name} generation failed: {e}"
            raise InferenceError(msg, backend=backend_name, cause=e) from e

    def _default_batch_generate(prompts: list[str]) -> list[str]:
        """Execute single-prompt generation across a batch of prompts."""
        return [_generate(p) for p in prompts]

    return create_common_app(
        backend_name=backend_name,
        model_name=model_name,
        startup_callback=startup_callback,
        generate_logic=_generate,
        batch_generate_logic=batch_generate_logic if batch_generate_logic is not None else _default_batch_generate,
        max_batch_size=max_batch_size,
    )


def serve_model_wrapper(
    backend_name: str,
    model_name: str,
    port: int,
    max_batch_size: int,
    app_factory: Callable[[], FastAPIAppProtocol],
    run_server: bool = False,
    host: str = "0.0.0.0",
    *,
    raise_on_error: bool | None = None,
    **kwargs: StatusValue,
) -> ServeResultDict:
    """Wrap serving logic to unify exception handling and result formatting.

    Args:
        backend_name: The name of the backend engine.
        model_name: Target model identifier.
        port: Listening network port.
        max_batch_size: Maximum concurrency/batch size.
        app_factory: Factory callback to create the FastAPI application.
        run_server: Whether to run the uvicorn HTTP server.
        host: Network host binding interface.
        raise_on_error: Whether to raise `ServingError` when `app_factory` or server startup fails.
        **kwargs: Additional keyword options inspected by `resolve_raise_on_error`.

    Returns:
        The execution status dictionary.

    Raises:
        DependencyMissingError: If FastAPI, uvicorn, or backend dependencies are missing.
        ServingError: If `raise_on_error` is enabled and server initialization fails.

    """
    if FastAPI is None or uvicorn is None:
        msg = "FastAPI and uvicorn are required for serving."
        raise DependencyMissingError(msg)

    should_raise = resolve_raise_on_error(raise_on_error=raise_on_error, kwargs=dict(kwargs))
    app = None
    error_message: str | None = None
    try:
        app = app_factory()
        status = f"running_{backend_name}_serve"
        status_code = ExecutionStatus.SUCCESS.value
        logger.info("Starting %s server on port %d", backend_name.title(), port)
        if run_server:
            assert isinstance(uvicorn, UvicornModuleProtocol)
            uvicorn.run(app, host=host, port=port)
    except (DependencyMissingError, ConfigurationError):
        raise
    except OPERATIONAL_EXCEPTIONS as e:
        logger.exception("Failed to start %s serve: ", backend_name)
        if should_raise:
            raise ServingError(
                f"Failed to start {backend_name} serve: {e!s}",
                backend=backend_name,
                model_name=model_name,
                operation="serve",
                cause=e,
            ) from e
        error_message = str(e)
        status = f"failed: {error_message}"
        status_code = ExecutionStatus.FAILED.value
        app = None

    result: ServeResultDict = {
        "backend": backend_name,
        "model": model_name,
        "port": port,
        "max_batch_size": max_batch_size,
        "status": status,
        "status_code": status_code,
        "mode": "continuous_batching",
        "app": app,
    }
    if error_message is not None:
        result["error_message"] = error_message
    return result


__all__ = [
    "DirectRunnableAppProtocol",
    "GenerateRequest",
    "JSONDict",
    "create_backend_serve_app",
    "create_common_app",
    "serve_model_wrapper",
]
