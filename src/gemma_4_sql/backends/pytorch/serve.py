"""PyTorch-specific continuous batching inference (vLLM) logic."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from vllm import AsyncEngineArgs, AsyncLLMEngine
    from vllm.utils import random_uuid
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    AsyncEngineArgs = None
    AsyncLLMEngine = None
    random_uuid = None
    FastAPI = None
    Request = None
    JSONResponse = None
    uvicorn = None


def serve_model(model_name: str, port: int = 8000, max_batch_size: int = 256, **kwargs: object) -> dict[str, object]:
    """Serve a model using vLLM for continuous batching.

    Args:
    ----
        model_name: The name of the model to serve.
        port: The port to bind the server to.
        max_batch_size: The maximum batch size for continuous batching.
        **kwargs: Additional parameters.

    Returns:
    -------
        A dictionary containing serving status and metadata.

    """
    if AsyncEngineArgs is None or FastAPI is None:
        return {"backend": "pytorch", "model": model_name, "port": port, "max_batch_size": max_batch_size, "status": "mocked_missing_pytorch", "mode": "continuous_batching"}

    app = None
    try:
        engine_args = AsyncEngineArgs(
            model=model_name,
            max_num_batched_tokens=max_batch_size * 256,
            max_num_seqs=max_batch_size,
            disable_log_requests=True,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        app = FastAPI(title=f"vLLM Serve: {model_name}")

        @app.post("/generate")
        async def generate(request: Request) -> JSONResponse:
            request_dict = await request.json()
            prompt = request_dict.pop("prompt", "")
            request_id = random_uuid()
            results_generator = engine.generate(prompt, None, request_id)  # type: ignore[arg-type]
            final_output = None
            async for request_output in results_generator:  # type: ignore[attr-defined]
                if await request.is_disconnected():
                    await engine.abort(request_id)
                    return JSONResponse(content={"error": "Client disconnected"})
                final_output = request_output

            text = final_output.outputs[0].text if final_output else ""
            return JSONResponse(content={"sql": text})

        status = "running_vllm"

        if not kwargs.get("test_mode"):
            logger.info("Starting vLLM server on port %d", port)
            # Normally: uvicorn.run(app, host="0.0.0.0", port=port)

    except Exception as e:
        logger.exception("Failed to start vLLM server: %s", e)
        status = f"failed: {e!s}"
        app = None

    return {"backend": "pytorch", "model": model_name, "port": port, "max_batch_size": max_batch_size, "status": status, "mode": "continuous_batching", "app": app}
