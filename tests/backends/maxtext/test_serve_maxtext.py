"""Tests for MaxText Serving."""

from unittest import mock

import pytest

import gemma_4_sql.backends.maxtext.serve as srv


def test_serve_model_maxtext_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(srv, "jax", None)
    res = srv.serve_model("foo")
    if not res["status"] == "mocked_missing_maxtext":
        raise AssertionError


def test_serve_model_maxtext_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(srv, "jax", object())
    monkeypatch.setattr(srv, "gemma4", object())
    monkeypatch.setattr(srv, "FastAPI", mock.MagicMock())
    monkeypatch.setattr(srv, "Request", mock.MagicMock())
    monkeypatch.setattr(srv, "JSONResponse", mock.MagicMock())
    monkeypatch.setattr(srv, "uvicorn", mock.MagicMock())
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["status"] == "running_maxtext_serve":
        raise AssertionError


def test_serve_model_maxtext_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(srv, "jax", object())
    monkeypatch.setattr(srv, "gemma4", object())

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function."""
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(srv, "FastAPI", raise_err)
    monkeypatch.setattr(srv, "uvicorn", mock.MagicMock())
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if "failed" not in str(res["status"]):
        raise AssertionError


@pytest.mark.asyncio
async def test_generate_endpoint() -> None:
    """Test generate endpoint logic directly."""
    importlib = __import__("importlib")
    importlib.reload(srv)
    srv.jax = object()
    srv.gemma4 = object()

    class MockJSONResponse:
        """Provide class docstring."""

        def __init__(self, content: dict) -> None:
            """Execute function."""
            self.content = content

    srv.JSONResponse = MockJSONResponse

    class MockApp:
        """Provide class docstring."""

        def post(self, *_args: object, **_kwargs: object) -> object:
            """Execute function."""

            def decorator(func: object) -> object:
                """Execute function."""
                self.func = func
                return func

            return decorator

    app_instance = MockApp()
    srv.FastAPI = lambda *_args, **_kwargs: app_instance
    srv.Request = mock.MagicMock()
    srv.uvicorn = mock.MagicMock()
    srv.serve_model("foo", test_mode=True)
    generate_func = app_instance.func
    request = mock.AsyncMock()
    request.json.return_value = {"prompt": "test"}
    result = await generate_func(request)
    if result.content["sql"] != "SELECT * FROM maxtext_serve WHERE prompt='test'":
        raise AssertionError


def test_serve_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib")
    sys = __import__("sys")
    m_serve = __import__("gemma_4_sql.backends.maxtext.serve")
    monkeypatch.setitem(sys.modules, "fastapi", None)
    importlib.reload(m_serve)
    monkeypatch.undo()
    importlib.reload(m_serve)


def test_serve_fastapi_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    m_serve = __import__("gemma_4_sql.backends.maxtext.serve")
    monkeypatch.setattr(m_serve, "FastAPI", None)
    monkeypatch.setattr(m_serve, "jax", object())
    monkeypatch.setattr(m_serve, "gemma4", object())
    res = m_serve.serve_model("m")
    if res["status"] != "failed_missing_fastapi":
        raise AssertionError
