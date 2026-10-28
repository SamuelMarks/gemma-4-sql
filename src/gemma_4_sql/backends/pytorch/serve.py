"""PyTorch-specific continuous batching inference (vLLM and native) logic."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from starlette.requests import Request
from starlette.responses import Response

from gemma_4_sql.backends.common_serve import (
    JSONResponseConstructorProtocol,
    create_backend_serve_app,
    serve_model_wrapper,
)
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import (
    FastAPIAppProtocol,
    JSONDict,
    JSONValue,
    ServeResultDict,
    VLLMEngineProtocol,
    VLLMRequestOutputProtocol,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _VLLMEngineArgsProtocol(Protocol):
    """Protocol for an initialized vLLM `AsyncEngineArgs` configuration instance."""


@runtime_checkable
class _VLLMEngineArgsConstructorProtocol(Protocol):
    """Protocol for `vllm.AsyncEngineArgs` constructor."""

    def __call__(
        self,
        *,
        model: str,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        disable_log_requests: bool,
    ) -> _VLLMEngineArgsProtocol:
        """Instantiate vLLM `AsyncEngineArgs`."""
        ...


@runtime_checkable
class _VLLMEngineLoaderProtocol(Protocol):
    """Protocol for `vllm.AsyncLLMEngine` class exposing `.from_engine_args(...)`."""

    def from_engine_args(self, engine_args: _VLLMEngineArgsProtocol | None, /) -> VLLMEngineProtocol:
        """Create a `VLLMEngineProtocol` from `engine_args`."""
        ...


@runtime_checkable
class _VLLMFastAPIConstructorProtocol(Protocol):
    """Protocol for `fastapi.FastAPI` constructor used by `_create_vllm_app`."""

    def __call__(self, *, title: str) -> FastAPIAppProtocol:
        """Construct a FastAPI application with `title`."""
        ...


torch = optional_import("torch")
FastAPI: _VLLMFastAPIConstructorProtocol | None = optional_attr(
    "fastapi",
    "FastAPI",
    _VLLMFastAPIConstructorProtocol,
)
JSONResponse: JSONResponseConstructorProtocol | None = optional_attr(
    "fastapi.responses",
    "JSONResponse",
    JSONResponseConstructorProtocol,
)
AsyncEngineArgs: _VLLMEngineArgsConstructorProtocol | None = optional_attr(
    "vllm",
    "AsyncEngineArgs",
    _VLLMEngineArgsConstructorProtocol,
)
AsyncLLMEngine: _VLLMEngineLoaderProtocol | None = optional_attr(
    "vllm",
    "AsyncLLMEngine",
    _VLLMEngineLoaderProtocol,
)
_raw_uuid = optional_attr("vllm.utils", "random_uuid")
random_uuid: Callable[[], str] | None = _raw_uuid if callable(_raw_uuid) else None  # ty: ignore[invalid-assignment]


def _create_vllm_app(model_name: str, max_batch_size: int) -> FastAPIAppProtocol | None:
    """Create the FastAPI application for the PyTorch vLLM server.

    Args:
        model_name: The name of the target model.
        max_batch_size: The maximum allowed batch size.

    Returns:
        The FastAPI application instance or `None` if FastAPI/JSONResponse is unavailable.

    """
    engine_args = (
        AsyncEngineArgs(
            model=model_name,
            max_num_batched_tokens=max_batch_size * 256,
            max_num_seqs=max_batch_size,
            disable_log_requests=True,
        )
        if AsyncEngineArgs is not None
        else None
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args) if AsyncLLMEngine is not None else None
    app = FastAPI(title=f"vLLM Serve: {model_name}") if FastAPI is not None else None

    if app is not None and JSONResponse is not None:
        json_resp_cls = JSONResponse

        @app.post("/generate")
        async def generate(request: Request) -> Response:
            """Execute vLLM-backed streaming request generation.

            Args:
                request: The incoming HTTP request.

            Returns:
                A JSON response containing the generated text.

            """
            assert engine is not None
            request_dict = await request.json()
            prompt = str(request_dict.pop("prompt", ""))
            request_id = random_uuid() if random_uuid is not None else ""
            results_generator = engine.generate(prompt, None, request_id)
            final_output: VLLMRequestOutputProtocol | None = None
            results_iterator = aiter(results_generator)
            try:
                while True:
                    request_output = await anext(results_iterator)
                    if await request.is_disconnected():
                        await engine.abort(request_id)
                        return json_resp_cls(content={"error": "Client disconnected"})  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-return-type]
                    final_output = request_output
            except StopAsyncIteration:
                pass
            text = final_output.outputs[0].text if final_output is not None and final_output.outputs else ""
            return json_resp_cls(content={"sql": text})  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-return-type]

        return app
    return None


_create_app = _create_vllm_app


def _create_native_app(model_name: str, max_batch_size: int) -> FastAPIAppProtocol:
    """Create native PyTorch in-process continuous batching FastAPI server.

    Args:
        model_name: Name of the model to serve.
        max_batch_size: Maximum continuous batch size.

    Returns:
        The FastAPI application instance.

    """

    def _call_pytorch_generate(model_name: str, prompt: str) -> JSONDict:
        """Delegate single-prompt generation to `pytorch.inference.generate_sql`."""
        from gemma_4_sql.backends.pytorch.inference import generate_sql

        res = generate_sql(model_name=model_name, prompt=prompt)
        return {k: v for k, v in res.items() if isinstance(v, (str, int, float, bool)) or v is None}

    return create_backend_serve_app(
        backend_name="pytorch",
        model_name=model_name,
        generate_sql_fn=_call_pytorch_generate,
        max_batch_size=max_batch_size,
    )


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, **kwargs: JSONValue) -> ServeResultDict:
    """Serve a model using vLLM or native PyTorch continuous batching.

    Args:
        model_name: The name of the target model.
        port: The network port to listen on.
        max_batch_size: The maximum allowed batch size.
        **kwargs: Underlying server and backend-specific configuration options.
                  Pass `native_fallback=True` or `engine='native'` for in-process continuous batching.

    Returns:
        A structured `ServeResultDict` containing the server metadata.

    Raises:
        DependencyMissingError: If vLLM dependencies are missing and native fallback is not active.

    """
    engine_type = str(kwargs.get("engine", "vllm" if not kwargs.get("native_fallback") else "native"))
    use_native = engine_type == "native" or bool(kwargs.get("native_fallback"))

    if not use_native and AsyncEngineArgs is None:
        raise DependencyMissingError("vLLM dependencies are missing for PyTorch serving.", backend="pytorch")

    if use_native:

        def app_factory() -> FastAPIAppProtocol:
            """Create FastAPI app backed by native PyTorch model inference."""
            return _create_native_app(model_name, max_batch_size)

    else:

        def app_factory() -> FastAPIAppProtocol:
            """Create FastAPI app backed by vLLM async engine."""
            vllm_app = _create_vllm_app(model_name, max_batch_size)
            assert vllm_app is not None
            return vllm_app

    result = serve_model_wrapper(
        backend_name="pytorch",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        app_factory=app_factory,
    )

    if not use_native:
        if result["status"] == "running_pytorch_serve":
            result["status"] = "running_vllm"

        if result["status"] == "running_vllm":
            logger.info("Starting vLLM server on port %d", port)
    else:
        if result["status"] == "running_pytorch_serve":
            logger.info("Starting native PyTorch server on port %d", port)

    return result
