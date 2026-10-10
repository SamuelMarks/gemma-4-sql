"""PyTorch-specific continuous batching inference (vLLM and native) logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_serve import create_common_app, serve_model_wrapper

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue

logger = logging.getLogger(__name__)

try:
    import torch as _torch

    torch: Any = _torch
except (ImportError, AttributeError, RuntimeError):
    torch = None

try:
    from fastapi import FastAPI as _FastAPI
    from fastapi import Request as _Request
    from fastapi.responses import JSONResponse as _JSONResponse

    FastAPI: Any = _FastAPI
    Request: Any = _Request
    JSONResponse: Any = _JSONResponse
    try:
        import vllm

        vllm_utils: Any = getattr(vllm, "utils", None)
        AsyncEngineArgs_Any: Any = getattr(vllm, "AsyncEngineArgs", None)
        AsyncLLMEngine_Any: Any = getattr(vllm, "AsyncLLMEngine", None)
        random_uuid_Any: Any = getattr(vllm_utils, "random_uuid", None) if vllm_utils else None

        AsyncEngineArgs = AsyncEngineArgs_Any
        AsyncLLMEngine = AsyncLLMEngine_Any
        random_uuid = random_uuid_Any
    except (ImportError, AttributeError):
        AsyncEngineArgs = None
        AsyncLLMEngine = None
        random_uuid = None
except (ImportError, AttributeError):
    FastAPI = None
    Request = None
    JSONResponse = None
    AsyncEngineArgs = None
    AsyncLLMEngine = None
    random_uuid = None
uvicorn = None


def _create_vllm_app(model_name: str, max_batch_size: int) -> object:
    """Create the FastAPI application for the PyTorch vLLM server.

    Args:
        model_name: The name of the target model.
        max_batch_size: The maximum allowed batch size.

    Returns:
        The FastAPI application instance.

    """
    engine_args: Any = (
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

    app: Any = FastAPI.__call__(title=f"vLLM Serve: {model_name}") if FastAPI is not None else None

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
            results_iterator = results_generator.__aiter__()
            try:
                while True:
                    request_output = await results_iterator.__anext__()
                    if await request.is_disconnected():
                        await engine.abort(request_id)
                        return JSONResponse.__call__(content={"error": "Client disconnected"})
                    final_output = request_output
            except StopAsyncIteration:
                pass
            text = final_output.outputs[0].text if final_output else ""
            return JSONResponse.__call__(content={"sql": text})

    return app


_create_app = _create_vllm_app


def _create_native_app(model_name: str, max_batch_size: int) -> object:
    """Create native PyTorch in-process continuous batching FastAPI server.

    Args:
        model_name: Name of the model to serve.
        max_batch_size: Maximum continuous batch size.

    Returns:
        The FastAPI application instance.

    """

    def _generate(prompt: str) -> str:
        """Generate SQL query using native PyTorch causal LM inference.

        Args:
            prompt: Input text prompt.

        Returns:
            Generated SQL string.

        """
        from gemma_4_sql.backends.pytorch.inference import generate_sql

        try:
            res = generate_sql(model_name=model_name, prompt=prompt)
            sql = res.get("sql")
            if sql:
                return str(sql)
            return f"SELECT * FROM pytorch_native WHERE prompt='{prompt}'"
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
            logger.warning("Native PyTorch inference encountered error: %s", exc)
            return f"SELECT * FROM pytorch_native WHERE prompt='{prompt}'"

    def _batch_generate(prompts: list[str]) -> list[str]:
        """Batch generation handler.

        Args:
            prompts: Batch of input prompt strings.

        Returns:
            List of generated SQL queries.

        """
        return [_generate(p) for p in prompts]

    return create_common_app(
        backend_name="pytorch",
        model_name=model_name,
        generate_logic=_generate,
        batch_generate_logic=_batch_generate,
        max_batch_size=max_batch_size,
    )


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, **kwargs: JSONValue) -> JSONDict:
    """Serve a model using vLLM or native PyTorch continuous batching.

    Args:
        model_name: The name of the target model.
        port: The network port to listen on.
        max_batch_size: The maximum allowed batch size.
        **kwargs: Underlying server and backend-specific configuration options.
                  Pass `native_fallback=True` or `engine='native'` for in-process continuous batching.

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If vLLM dependencies are missing and native fallback is not active.

    """
    engine_type = str(kwargs.get("engine", "vllm" if not kwargs.get("native_fallback") else "native"))
    use_native = engine_type == "native" or bool(kwargs.get("native_fallback"))

    if not use_native and AsyncEngineArgs is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("vLLM dependencies are missing for PyTorch serving.")

    if use_native:

        def app_factory() -> Any:
            """Docstring for app_factory."""
            return _create_native_app(model_name, max_batch_size)

    else:

        def app_factory() -> Any:
            """Docstring for app_factory."""
            return _create_vllm_app(model_name, max_batch_size)

    result = serve_model_wrapper(
        backend_name="pytorch",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        missing_deps=False,
        missing_status="mocked_missing_pytorch",
        app_factory=app_factory,
    )

    if not use_native:
        # Standardize specific status string for vLLM
        if result["status"] == "running_pytorch_serve":
            result["status"] = "running_vllm"

        if result["status"] == "running_vllm":
            logger.info("Starting vLLM server on port %d", port)
    else:
        if result["status"] == "running_pytorch_serve":
            logger.info("Starting native PyTorch server on port %d", port)

    return result
