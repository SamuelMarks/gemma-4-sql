"""PyTorch-specific continuous batching inference (vLLM) logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_serve import serve_model_wrapper
from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)
AsyncEngineArgs = None
AsyncLLMEngine = None
random_uuid = None
FastAPI = None
Request = None
JSONResponse = None
uvicorn = None
with catch_optional_imports():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from vllm import AsyncEngineArgs, AsyncLLMEngine
    from vllm.utils import random_uuid


def _create_app(model_name: str, max_batch_size: int) -> object:
    """Create the FastAPI application for the PyTorch vLLM server.

    Args:
        model_name: The name of the target model.
        max_batch_size: The maximum allowed batch size.

    Returns:
        The execution result.
    """
    engine_args = AsyncEngineArgs(model=model_name, max_num_batched_tokens=max_batch_size * 256, max_num_seqs=max_batch_size, disable_log_requests=True)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    app = FastAPI(title=f"vLLM Serve: {model_name}")

    @app.post("/generate")
    async def generate(request: Request) -> JSONResponse:
        """Execute logic.

        Returns:
            object: The resulting output from the operation.

        """
        request_dict = await request.json()
        prompt = request_dict.pop("prompt", "")
        request_id = random_uuid()
        results_generator = engine.generate(prompt, None, request_id)
        final_output = None
        async for request_output in results_generator:  # pragma: no cover
            if await request.is_disconnected():
                await engine.abort(request_id)
                return JSONResponse(content={"error": "Client disconnected"})
            final_output = request_output
        text = final_output.outputs[0].text if final_output else ""
        return JSONResponse(content={"sql": text})

    return app


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, **kwargs: JSONValue) -> JSONDict:
    """Serve a model using vLLM for continuous batching.

        Args:
                    **kwargs: Underlying server and backend-specific configuration options.
    model_name: The name of the target model.
            port: The network port to listen on.
            max_batch_size: The maximum allowed batch size.

        Returns:
            A dictionary containing the results.
    """
    if AsyncEngineArgs is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("vLLM dependencies are missing for PyTorch serving.")

    result = serve_model_wrapper(
        backend_name="pytorch",
        model_name=model_name,
        port=port,
        max_batch_size=max_batch_size,
        missing_deps=False,
        missing_status="mocked_missing_pytorch",
        app_factory=lambda: _create_app(model_name, max_batch_size),
        test_mode=bool(kwargs.get("test_mode")),
    )

    # Standardize specific status string
    if result["status"] == "running_pytorch_serve":
        result["status"] = "running_vllm"

    # Maintain specific log from original code
    if result["status"] == "running_vllm" and not kwargs.get("test_mode"):
        logger.info("Starting vLLM server on port %d", port)

    return result
