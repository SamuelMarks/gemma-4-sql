# Copyright 2024
"""Tests for MaxText Serving."""

from unittest import mock

import pytest

import gemma_4_sql.backends.maxtext.serve as srv


def test_serve_model_maxtext_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(srv, "jax", None)
    res = srv.serve_model("foo")
    if not res["status"] == "mocked_missing_maxtext":
        raise AssertionError


def test_serve_model_maxtext_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(srv, "jax", object())
    monkeypatch.setattr(srv, "gemma4", object())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", mock.MagicMock())
    monkeypatch.setattr(srv, "Request", mock.MagicMock())
    monkeypatch.setattr(srv, "JSONResponse", mock.MagicMock())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.uvicorn", mock.MagicMock())
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["status"] == "running_maxtext_serve":
        raise AssertionError


def test_serve_model_maxtext_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(srv, "jax", object())
    monkeypatch.setattr(srv, "gemma4", object())

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", raise_err)
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.uvicorn", mock.MagicMock())
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if "failed" not in str(res["status"]):
        raise AssertionError


@pytest.mark.asyncio
async def test_generate_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate endpoint logic directly.

    Raises:
        AssertionError: Description.

    """
    importlib = __import__("importlib", fromlist=[""])
    importlib.reload(srv)
    srv.jax = object()
    srv.gemma4 = object()

    class MockJSONResponse:
        def __init__(self, content: dict) -> None:
            self.body = str(content).encode()

    monkeypatch.setattr("gemma_4_sql.backends.common_serve.JSONResponse", MockJSONResponse)

    mock_app = mock.MagicMock()
    mock_app.router.routes = []

    def mock_post(*_args: object, **_kwargs: object) -> object:
        def decorator(func: object) -> object:
            route = mock.MagicMock()
            route.endpoint = func
            mock_app.router.routes.append(route)
            return func

        return decorator

    mock_app.post = mock_post
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", lambda *_args, **_kwargs: mock_app)
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.Request", mock.MagicMock())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.uvicorn", mock.MagicMock())
    srv.serve_model("foo", test_mode=True)
    generate_func = mock_app.router.routes[-1].endpoint
    request = mock.AsyncMock()
    request.json.return_value = {"prompt": "test"}
    result = await generate_func(request)
    sql_val = result.body.decode() if hasattr(result, "body") else ""
    if "SELECT * FROM maxtext_serve WHERE prompt='test'" not in sql_val:
        raise AssertionError


def test_serve_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    m_serve = __import__("gemma_4_sql.backends.maxtext.serve", fromlist=[""])
    monkeypatch.setitem(sys.modules, "fastapi", None)
    importlib.reload(m_serve)
    monkeypatch.undo()
    importlib.reload(m_serve)


def test_serve_fastapi_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    m_serve = __import__("gemma_4_sql.backends.maxtext.serve", fromlist=[""])
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", None)
    monkeypatch.setattr(m_serve, "jax", object())
    monkeypatch.setattr(m_serve, "gemma4", object())
    res = m_serve.serve_model("m")
    if res["status"] != "failed_missing_fastapi":
        raise AssertionError
