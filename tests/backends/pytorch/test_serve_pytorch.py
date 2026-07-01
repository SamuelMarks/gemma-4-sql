"""Tests for PyTorch Serving."""

from __future__ import annotations

import typing
from unittest import mock

import pytest

import gemma_4_sql.backends.pytorch.serve as srv


class MockAsyncEngineArgs:
    def __init__(self, **kwargs: object) -> None:
        pass


class MockAsyncLLMEngine:
    @staticmethod
    def from_engine_args(args: object) -> object:
        class Engine:
            def generate(self, *args: object, **kwargs: object) -> object:
                class Output:
                    class Out:
                        text = "SELECT * FROM vllm"

                    outputs = [Out()]

                async def gen() -> typing.AsyncGenerator:
                    yield Output()

                return gen()

            async def abort(self, req_id: object) -> None:
                pass

        return Engine()


def mock_random_uuid() -> str:
    return "123"


def test_serve_model_pytorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(srv, "AsyncEngineArgs", None)
    res = srv.serve_model("foo")
    if not res["status"] == "mocked_missing_pytorch":
        raise AssertionError


def test_serve_model_pytorch_real(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(srv, "AsyncEngineArgs", MockAsyncEngineArgs)
    monkeypatch.setattr(srv, "AsyncLLMEngine", MockAsyncLLMEngine)
    monkeypatch.setattr(srv, "random_uuid", mock_random_uuid)
    monkeypatch.setattr(srv, "FastAPI", mock.MagicMock())
    monkeypatch.setattr(srv, "Request", mock.MagicMock())
    monkeypatch.setattr(srv, "JSONResponse", mock.MagicMock())
    monkeypatch.setattr(srv, "uvicorn", mock.MagicMock())

    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["status"] == "running_vllm":
        raise AssertionError
    if not res["port"] == 8000:
        raise AssertionError


def test_serve_model_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(srv, "AsyncEngineArgs", MockAsyncEngineArgs)
    monkeypatch.setattr(srv, "AsyncLLMEngine", MockAsyncLLMEngine)
    monkeypatch.setattr(srv, "FastAPI", mock.MagicMock())

    def raise_err(*args: object, **kwargs: object) -> object:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockAsyncLLMEngine, "from_engine_args", raise_err)

    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if "failed" not in str(res["status"]):
        raise AssertionError


@pytest.mark.asyncio()
async def test_generate_endpoint() -> None:
    """Test generate endpoint logic directly."""
    import importlib

    importlib.reload(srv)
    srv.AsyncEngineArgs = MockAsyncEngineArgs  # type: ignore[misc]
    srv.AsyncLLMEngine = MockAsyncLLMEngine  # type: ignore[misc]
    srv.random_uuid = mock_random_uuid  # type: ignore[misc]

    class MockJSONResponse:
        def __init__(self, content: dict) -> None:
            self.content = content

    srv.JSONResponse = MockJSONResponse  # type: ignore[misc]

    class MockApp:
        def post(self, *args, **kwargs):
            def decorator(func):
                self.func = func
                return func

            return decorator

    app_instance = MockApp()
    srv.FastAPI = lambda *args, **kwargs: app_instance

    srv.Request = mock.MagicMock()

    res = srv.serve_model("foo", test_mode=True)
    res["app"]

    generate_func = app_instance.func

    request = mock.AsyncMock()
    request.json.return_value = {"prompt": "test"}
    request.is_disconnected.return_value = False

    result = await generate_func(request)
    assert result.content["sql"] == "SELECT * FROM vllm"

    # Test disconnect
    request.is_disconnected.return_value = True
    result2 = await generate_func(request)
    assert "error" in result2.content


def test_serve_imports_success(monkeypatch: pytest.MonkeyPatch):
    import importlib
    import sys

    import gemma_4_sql.backends.pytorch.serve as m_serve

    monkeypatch.setitem(sys.modules, "uvicorn", type("M", (), {})())
    monkeypatch.setitem(sys.modules, "fastapi", type("M", (), {"FastAPI": None, "Request": None})())
    monkeypatch.setitem(sys.modules, "fastapi.responses", type("M", (), {"JSONResponse": None})())
    monkeypatch.setitem(sys.modules, "vllm", type("M", (), {"AsyncEngineArgs": None, "AsyncLLMEngine": None})())
    monkeypatch.setitem(sys.modules, "vllm.utils", type("M", (), {"random_uuid": None})())

    importlib.reload(m_serve)
    monkeypatch.undo()
    importlib.reload(m_serve)
