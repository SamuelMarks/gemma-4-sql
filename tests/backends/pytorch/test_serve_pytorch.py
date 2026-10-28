"""Tests for PyTorch continuous batching and vLLM serving."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import requires_torch

pytestmark = requires_torch

import pytest

from gemma_4_sql.backends.common_serve import DirectRunnableAppProtocol
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError
from gemma_4_sql.type_hints import FastAPIAppProtocol, JSONDict

type RouteHandler = Callable[[MagicMock], Awaitable[JSONDict]]


@pytest.fixture(autouse=True)
def _restore_serve_mod(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore serve_module globals after each test."""
    import gemma_4_sql.backends.pytorch.serve as serve_module

    monkeypatch.setattr(serve_module, "torch", serve_module.torch)
    monkeypatch.setattr(serve_module, "FastAPI", serve_module.FastAPI)
    monkeypatch.setattr(serve_module, "JSONResponse", serve_module.JSONResponse)
    monkeypatch.setattr(serve_module, "AsyncEngineArgs", serve_module.AsyncEngineArgs)
    monkeypatch.setattr(serve_module, "AsyncLLMEngine", serve_module.AsyncLLMEngine)


@pytest.mark.asyncio
async def test_create_vllm_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _create_vllm_app registers /generate, streams vLLM outputs, and handles disconnects."""
    import gemma_4_sql.backends.pytorch.serve as serve_module

    mock_app = MagicMock()
    mock_fastapi = MagicMock(return_value=mock_app)

    monkeypatch.setattr(serve_module, "FastAPI", mock_fastapi)
    monkeypatch.setattr(serve_module, "AsyncEngineArgs", MagicMock())
    monkeypatch.setattr(serve_module, "AsyncLLMEngine", MagicMock())
    monkeypatch.setattr(serve_module, "JSONResponse", MagicMock(side_effect=lambda content: {"response": content}))  # pyright: ignore[reportUnknownLambdaType]

    def mock_decorator(_path: str) -> Callable[[RouteHandler], RouteHandler]:
        """Capture registered route function."""

        def wrapper(func: RouteHandler) -> RouteHandler:
            """Store route handler on mock_app."""
            mock_app.generate_func = func
            return func

        return wrapper

    mock_app.post.side_effect = mock_decorator

    mock_req = MagicMock()
    mock_req.json = AsyncMock(return_value={"prompt": "p"})
    mock_req.is_disconnected = AsyncMock(return_value=False)

    mock_engine = MagicMock()
    mock_async_llm = MagicMock()
    mock_async_llm.from_engine_args.return_value = mock_engine
    monkeypatch.setattr(serve_module, "AsyncLLMEngine", mock_async_llm)

    class MockOutput:
        """Simulated vLLM request output."""

        class Out:
            """Simulated completion output."""

            text = "sql"

        outputs = [Out()]

    async def mock_generate_stream(*_args: str, **_kwargs: str) -> AsyncIterator[MockOutput]:
        """Yield a single MockOutput."""
        yield MockOutput()

    mock_engine.generate.return_value = mock_generate_stream()

    app = serve_module._create_vllm_app("m", 1)
    assert app == mock_app
    generate_fn: RouteHandler = mock_app.generate_func

    res = await generate_fn(mock_req)
    assert res == {"response": {"sql": "sql"}}

    # Client disconnect aborts vLLM request
    mock_req.is_disconnected = AsyncMock(return_value=True)
    mock_engine.abort = AsyncMock()
    mock_engine.generate.return_value = mock_generate_stream()

    res2 = await generate_fn(mock_req)
    assert res2 == {"response": {"error": "Client disconnected"}}
    mock_engine.abort.assert_awaited_once()

    # Missing JSONResponse returns None
    serve_module.JSONResponse = None
    assert serve_module._create_vllm_app("m", 1) is None


def test_create_native_app() -> None:
    """Verify _create_native_app builds a real FastAPI app and serves /health and /generate via TestClient."""
    from fastapi.testclient import TestClient

    import gemma_4_sql.backends.pytorch.serve as serve_module

    app = serve_module._create_native_app("m", 16)
    assert isinstance(app, DirectRunnableAppProtocol)
    with patch("gemma_4_sql.backends.pytorch.inference.generate_sql") as mock_gen:
        mock_gen.return_value = {"sql": "SELECT 1;"}
        assert app.run_direct("p") == "SELECT 1;"

        with TestClient(app) as client:
            health_res = client.get("/health")
            assert health_res.status_code == 200
            assert health_res.json()["backend"] == "pytorch"

            gen_res = client.post("/generate", json={"prompt": "SELECT 1"})
            assert gen_res.status_code == 200
            assert gen_res.json()["sql"] == "SELECT 1;"

        mock_gen.return_value = {}
        with pytest.raises(InferenceError, match="PyTorch inference returned empty SQL"):
            app.run_direct("p")

        mock_gen.side_effect = RuntimeError("error")
        with pytest.raises(InferenceError, match="PyTorch generation failed"):
            app.run_direct("p")


def test_serve_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify serve_model dispatches to vLLM or native PyTorch continuous batching through the real serve_model_wrapper."""
    from fastapi.testclient import TestClient

    import gemma_4_sql.backends.pytorch.serve as serve_module

    monkeypatch.setattr(serve_module, "AsyncEngineArgs", MagicMock())
    monkeypatch.setattr(serve_module, "AsyncLLMEngine", MagicMock())

    res_vllm = serve_module.serve_model("m")
    assert res_vllm["status"] == "running_vllm"
    assert res_vllm.get("app") is not None

    res_native = serve_module.serve_model("m", native_fallback=True)
    assert res_native["status"] == "running_pytorch_serve"
    native_app = res_native.get("app")
    assert isinstance(native_app, FastAPIAppProtocol)
    with patch("gemma_4_sql.backends.pytorch.inference.generate_sql", return_value={"sql": "SELECT 42;"}):
        with TestClient(native_app) as client:
            resp = client.post("/generate", json={"prompt": "test"})
            assert resp.status_code == 200
            assert resp.json()["sql"] == "SELECT 42;"

    with patch("gemma_4_sql.backends.pytorch.serve._create_native_app", side_effect=RuntimeError("native boom")):
        res_fail_native = serve_module.serve_model("m", native_fallback=True)
        assert res_fail_native["status"] == "failed: native boom"

    with patch("gemma_4_sql.backends.pytorch.serve._create_vllm_app", side_effect=RuntimeError("vllm boom")):
        res_fail_vllm = serve_module.serve_model("m")
        assert res_fail_vllm["status"] == "failed: vllm boom"

    serve_module.AsyncEngineArgs = None
    with pytest.raises(DependencyMissingError):
        serve_module.serve_model("m")


@pytest.mark.asyncio
async def test_create_vllm_app_empty_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _create_vllm_app returns empty SQL when the vLLM stream yields zero items."""
    import gemma_4_sql.backends.pytorch.serve as serve_module

    mock_app = MagicMock()
    monkeypatch.setattr(serve_module, "FastAPI", MagicMock(return_value=mock_app))
    monkeypatch.setattr(serve_module, "AsyncEngineArgs", MagicMock())
    monkeypatch.setattr(serve_module, "AsyncLLMEngine", MagicMock())
    monkeypatch.setattr(serve_module, "JSONResponse", MagicMock(side_effect=lambda content: {"response": content}))  # pyright: ignore[reportUnknownLambdaType]

    def mock_decorator(_path: str) -> Callable[[RouteHandler], RouteHandler]:
        """Capture route handler."""

        def wrapper(func: RouteHandler) -> RouteHandler:
            """Store route handler on mock_app."""
            mock_app.generate_func = func
            return func

        return wrapper

    class EmptyAsyncGen:
        """Async iterator yielding zero items."""

        def __aiter__(self) -> EmptyAsyncGen:
            """Return self."""
            return self

        async def __anext__(self) -> None:
            """Stop immediately."""
            raise StopAsyncIteration

    mock_engine = MagicMock()
    mock_engine.generate.return_value = EmptyAsyncGen()
    mock_async_llm2 = MagicMock()
    mock_async_llm2.from_engine_args.return_value = mock_engine
    monkeypatch.setattr(serve_module, "AsyncLLMEngine", mock_async_llm2)

    mock_app.post.side_effect = mock_decorator
    serve_module._create_vllm_app("m", 1)
    generate_fn: RouteHandler = mock_app.generate_func

    mock_req = MagicMock()
    mock_req.json = AsyncMock(return_value={"prompt": "p"})
    mock_req.is_disconnected = AsyncMock(return_value=False)

    res = await generate_fn(mock_req)
    assert res == {"response": {"sql": ""}}
