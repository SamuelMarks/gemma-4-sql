# Copyright 2024
"""Tests for PyTorch Serving."""

from __future__ import annotations

import typing
from unittest import mock

import pytest

import gemma_4_sql.backends.pytorch.serve as srv


class MockAsyncEngineArgs:
    """Provide class docstring."""

    def __init__(self, **kwargs: object) -> None:
        """Execute function."""


class MockAsyncLLMEngine:
    """Provide class docstring."""

    @staticmethod
    def from_engine_args(_args: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """

        class Engine:
            """Provide class docstring."""

            def generate(self, *_args: object, **_kwargs: object) -> object:
                """Execute function.

                Returns:
                    object: Description of return.

                """

                class Output:
                    """Provide class docstring."""

                    class Out:
                        """Provide class docstring."""

                        text = "SELECT * FROM vllm"

                    outputs: typing.ClassVar = [Out()]

                async def gen() -> typing.AsyncGenerator:
                    """Execute function."""
                    yield Output()

                return gen()

            async def abort(self, req_id: object) -> None:
                """Execute function."""

        return Engine()


def mock_random_uuid() -> str:
    """Execute function.

    Returns:
        object: Description of return.

    """
    return "123"


def test_serve_model_pytorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(srv, "AsyncEngineArgs", None)
    res = srv.serve_model("foo")
    if not res["status"] == "mocked_missing_pytorch":
        raise AssertionError


def test_serve_model_pytorch_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(srv, "AsyncEngineArgs", MockAsyncEngineArgs)
    monkeypatch.setattr(srv, "AsyncLLMEngine", MockAsyncLLMEngine)
    monkeypatch.setattr(srv, "random_uuid", mock_random_uuid)

    class MockFastAPI:
        """Docstring."""

        def __init__(self, **_kwargs: object) -> None:
            """Docstring."""
            self.router = mock.MagicMock()
            self.func = None

        def post(self, *_args: object, **_kwargs: object) -> object:
            """Docstring."""

            def decorator(func: object) -> object:
                """Docstring."""
                self.func = func
                return func

            return decorator

    monkeypatch.setattr(srv, "FastAPI", MockFastAPI)
    monkeypatch.setattr(srv, "Request", mock.MagicMock())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.uvicorn", mock.MagicMock())
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["status"] == "running_vllm":
        raise AssertionError
    if not res["port"] == int("8000"):
        raise AssertionError


def test_serve_model_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(srv, "AsyncEngineArgs", MockAsyncEngineArgs)
    monkeypatch.setattr(srv, "AsyncLLMEngine", MockAsyncLLMEngine)

    class MockFastAPI:
        """Docstring."""

        def __init__(self, **_kwargs: object) -> None:
            """Docstring."""
            self.router = mock.MagicMock()
            self.func = None

        def post(self, *_args: object, **_kwargs: object) -> object:
            """Docstring."""

            def decorator(func: object) -> object:
                """Docstring."""
                self.func = func
                return func

            return decorator

    monkeypatch.setattr(srv, "FastAPI", MockFastAPI)

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockAsyncLLMEngine, "from_engine_args", raise_err)
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if "failed" not in str(res["status"]):
        raise AssertionError


@pytest.mark.asyncio
async def test_generate_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate endpoint logic directly.

    Raises:
        AssertionError: Description.

    """
    __import__("importlib", fromlist=[""])

    srv.AsyncEngineArgs = MockAsyncEngineArgs
    srv.AsyncLLMEngine = MockAsyncLLMEngine
    srv.random_uuid = mock_random_uuid

    class MockJSONResponse:
        """Provide class docstring."""

        def __init__(self, content: dict) -> None:
            """Execute function."""
            self.content = content

    srv.JSONResponse = MockJSONResponse

    class MockApp:
        """Provide class docstring."""

        def __init__(self) -> None:
            """Execute function."""
            self.func = None

        def post(self, *_args: object, **_kwargs: object) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """

            def decorator(func: object) -> object:
                """Execute function.

                Returns:
                    object: Description of return.

                """
                self.func = func
                return func

            return decorator

    app_instance = MockApp()
    srv.FastAPI = lambda *_args, **_kwargs: app_instance
    srv.Request = mock.MagicMock()
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", lambda *_args, **_kwargs: app_instance)
    monkeypatch.setattr(srv, "Request", mock.MagicMock())
    import gemma_4_sql.backends.common_serve

    if hasattr(gemma_4_sql.backends.common_serve, "Request"):
        monkeypatch.setattr("gemma_4_sql.backends.common_serve.Request", mock.MagicMock())
    res = srv.serve_model("foo", test_mode=True)
    res["app"]
    generate_func = app_instance.func
    request = mock.AsyncMock()
    request.json.return_value = {"prompt": "test"}
    request.is_disconnected.return_value = False
    result = await generate_func(request)
    if result.content["sql"] != "SELECT * FROM vllm":
        raise AssertionError
    request.is_disconnected.return_value = True
    result2 = await generate_func(request)
    if "error" not in result2.content:
        raise AssertionError


def test_serve_imports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    m_serve = __import__("gemma_4_sql.backends.pytorch.serve", fromlist=[""])
    monkeypatch.setitem(sys.modules, "uvicorn", type("M", (), {})())
    monkeypatch.setitem(sys.modules, "fastapi", type("M", (), {"FastAPI": None, "Request": None})())
    monkeypatch.setitem(sys.modules, "fastapi.responses", type("M", (), {"JSONResponse": None})())
    monkeypatch.setitem(sys.modules, "vllm", type("M", (), {"AsyncEngineArgs": None, "AsyncLLMEngine": None})())
    monkeypatch.setitem(sys.modules, "vllm.utils", type("M", (), {"random_uuid": None})())
    importlib.reload(m_serve)
    monkeypatch.undo()
    importlib.reload(m_serve)
