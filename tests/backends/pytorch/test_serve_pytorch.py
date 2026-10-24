"""Tests for PyTorch continuous batching and vLLM serving."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import requires_torch

pytestmark = requires_torch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


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
    monkeypatch.setattr(serve_module, "JSONResponse", MagicMock(side_effect=lambda content: {"response": content}))

    def mock_decorator(_path: Any) -> Any:
        """Capture registered route function."""

        def wrapper(func: Any) -> Any:
            """Store route handler on mock_app."""
            mock_app.generate_func = func
            return func

        return wrapper

    mock_app.post.side_effect = mock_decorator

    mock_req = MagicMock()
    mock_req.json = AsyncMock(return_value={"prompt": "p"})
    mock_req.is_disconnected = AsyncMock(return_value=False)

    mock_engine = MagicMock()
    serve_module.AsyncLLMEngine.from_engine_args.return_value = mock_engine

    class MockOutput:
        """Simulated vLLM request output."""

        class Out:
            """Simulated completion output."""

            text = "sql"

        outputs = [Out()]

    async def mock_generate_stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[MockOutput]:
        """Yield a single MockOutput."""
        yield MockOutput()

    mock_engine.generate.return_value = mock_generate_stream()

    app = serve_module._create_vllm_app("m", 1)
    assert app == mock_app
    generate_fn = mock_app.generate_func

    res = await generate_fn(mock_req)
    assert res == {"response": {"sql": "sql"}}

    # Client disconnect aborts vLLM request
    mock_req.is_disconnected = AsyncMock(return_value=True)
    mock_engine.abort = AsyncMock()
    mock_engine.generate.return_value = mock_generate_stream()

    res2 = await generate_fn(mock_req)
    assert res2 == {"response": {"error": "Client disconnected"}}
    mock_engine.abort.assert_awaited_once()

    # Missing JSONResponse returnsNone
    serve_module.JSONResponse = None
    assert serve_module._create_vllm_app("m", 1) is None


def test_create_native_app() -> None:
    """Verify _create_native_app delegates to create_common_app and handles SQL generation."""
    import gemma_4_sql.backends.pytorch.serve as serve_module

    with patch("gemma_4_sql.backends.common_serve.create_common_app") as mock_create:
        serve_module._create_native_app("m", 16)
        mock_create.assert_called_once()
        assert mock_create.call_args[1]["max_batch_size"] == 16

        generate_logic = mock_create.call_args[1]["generate_logic"]
        batch_generate_logic = mock_create.call_args[1]["batch_generate_logic"]

        with patch("gemma_4_sql.backends.pytorch.inference.generate_sql") as mock_gen:
            mock_gen.return_value = {"sql": "SELECT 1;"}
            assert generate_logic("p") == "SELECT 1;"

            mock_gen.return_value = {}
            with pytest.raises(InferenceError, match="PyTorch inference returned empty SQL"):
                generate_logic("p")

            mock_gen.side_effect = RuntimeError("error")
            with pytest.raises(InferenceError, match="PyTorch generation failed"):
                generate_logic("p")

        with patch("gemma_4_sql.backends.pytorch.inference.generate_sql") as mock_gen:
            mock_gen.return_value = {"sql": "sql"}
            assert batch_generate_logic(["p1", "p2"]) == ["sql", "sql"]


def test_serve_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify serve_model dispatches to vLLM or native PyTorch continuous batching."""
    import gemma_4_sql.backends.pytorch.serve as serve_module

    monkeypatch.setattr(serve_module, "AsyncEngineArgs", MagicMock())

    with patch("gemma_4_sql.backends.pytorch.serve.serve_model_wrapper") as mock_wrapper:
        mock_wrapper.return_value = {"status": "running_pytorch_serve"}
        res = serve_module.serve_model("m")
        assert res["status"] == "running_vllm"

        app_factory = mock_wrapper.call_args[1]["app_factory"]
        with patch("gemma_4_sql.backends.pytorch.serve._create_vllm_app") as mock_create_vllm:
            app_factory()
            mock_create_vllm.assert_called_once()

        mock_wrapper.return_value = {"status": "running_pytorch_serve"}
        res = serve_module.serve_model("m", native_fallback=True)
        assert res["status"] == "running_pytorch_serve"

        app_factory_native = mock_wrapper.call_args[1]["app_factory"]
        with patch("gemma_4_sql.backends.pytorch.serve._create_native_app") as mock_create_native:
            app_factory_native()
            mock_create_native.assert_called_once()

        mock_wrapper.return_value = {"status": "failed"}
        res = serve_module.serve_model("m", native_fallback=True)
        assert res["status"] == "failed"

        mock_wrapper.return_value = {"status": "failed"}
        res = serve_module.serve_model("m")
        assert res["status"] == "failed"

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
    monkeypatch.setattr(serve_module, "JSONResponse", MagicMock(side_effect=lambda content: {"response": content}))

    def mock_decorator(_path: Any) -> Any:
        """Capture route handler."""

        def wrapper(func: Any) -> Any:
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
    serve_module.AsyncLLMEngine.from_engine_args.return_value = mock_engine

    mock_app.post.side_effect = mock_decorator
    serve_module._create_vllm_app("m", 1)
    generate_fn = mock_app.generate_func

    mock_req = MagicMock()
    mock_req.json = AsyncMock(return_value={"prompt": "p"})
    mock_req.is_disconnected = AsyncMock(return_value=False)

    res = await generate_fn(mock_req)
    assert res == {"response": {"sql": ""}}
