"""PyTorch-specific continuous batching inference (vLLM and native) logic."""

from __future__ import annotations

import logging
from typing import Any

from gemma_4_sql.backends.common_serve import create_backend_serve_app, serve_model_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import JSONDict, JSONValue, ServeResultDict

logger = logging.getLogger(__name__)

torch = optional_import("torch")
FastAPI = optional_attr("fastapi", "FastAPI")
JSONResponse = optional_attr("fastapi.responses", "JSONResponse")
AsyncEngineArgs = optional_attr("vllm", "AsyncEngineArgs")
AsyncLLMEngine = optional_attr("vllm", "AsyncLLMEngine")
random_uuid = optional_attr("vllm.utils", "random_uuid")


def _create_vllm_app(model_name: str, max_batch_size: int) -> object:
    """Create the FastAPI application for the PyTorch vLLM server.

    Args:
        model_name: The name of the target model.
        max_batch_size: The maximum allowed batch size.

    Returns:
        The FastAPI application instance.

    """
    engine_args = (
        AsyncEngineArgs.__call__(
            model=model_name,
            max_num_batched_tokens=max_batch_size * 256,
            max_num_seqs=max_batch_size,
            disable_log_requests=True,
        )
        if AsyncEngineArgs is not None
        else None
    )
    engine: Any = AsyncLLMEngine.from_engine_args(engine_args) if AsyncLLMEngine is not None else None

    app = FastAPI.__call__(title=f"vLLM Serve: {model_name}") if FastAPI is not None else None

    if app is not None and JSONResponse is not None:

        @app.post("/generate")
        async def generate(request: Any) -> Any:
            """Execute vLLM-backed streaming request generation.

            Args:
                request: The incoming HTTP request.

            Returns:
                A JSON response containing the generated text.

            """
            request_dict = await request.json()
            prompt = request_dict.pop("prompt", "")
            request_id = random_uuid() if random_uuid is not None else ""
            results_generator = engine.generate(prompt, None, request_id)
            final_output = None
            results_iterator = aiter(results_generator)
            try:
                while True:
                    request_output = await anext(results_iterator)
                    if await request.is_disconnected():
                        await engine.abort(request_id)
                        return JSONResponse.__call__(content={"error": "Client disconnected"})
                    final_output = request_output
            except StopAsyncIteration:
                pass
            text = final_output.outputs[0].text if final_output else ""
            return JSONResponse.__call__(content={"sql": text})

        return app
    return None


_create_app = _create_vllm_app


def _create_native_app(model_name: str, max_batch_size: int) -> object:
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

        return dict(generate_sql(model_name=model_name, prompt=prompt))

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

        def app_factory() -> Any:
            """Create FastAPI app backed by native PyTorch model inference."""
            return _create_native_app(model_name, max_batch_size)

    else:

        def app_factory() -> Any:
            """Create FastAPI app backed by vLLM async engine."""
            return _create_vllm_app(model_name, max_batch_size)

    result = serve_model_wrapper(
        backend_name="pytorch",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        missing_deps=False,
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
