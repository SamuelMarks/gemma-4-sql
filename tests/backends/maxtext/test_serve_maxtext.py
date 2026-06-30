"""Tests for MaxText Serving."""

from unittest import mock

import pytest

import gemma_4_sql.backends.maxtext.serve as srv


def test_serve_model_maxtext_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(srv, "jax", None)
    res = srv.serve_model("foo")
    if not res["status"] == "mocked_missing_maxtext":
        raise AssertionError


def test_serve_model_maxtext_real(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(srv, "jax", object())
    monkeypatch.setattr(srv, "gemma4", object())

    def raise_err(*args: object, **kwargs: object) -> object:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(srv, "FastAPI", raise_err)
    monkeypatch.setattr(srv, "uvicorn", mock.MagicMock())

    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if "failed" not in str(res["status"]):
        raise AssertionError


@pytest.mark.asyncio()
async def test_generate_endpoint() -> None:
    """Test generate endpoint logic directly."""
    import importlib

    importlib.reload(srv)
    srv.jax = object()
    srv.gemma4 = object()

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
    srv.FastAPI = lambda *args, **kwargs: app_instance  # type: ignore[misc]
    srv.Request = mock.MagicMock()  # type: ignore[misc]
    srv.uvicorn = mock.MagicMock()  # type: ignore[misc]

    srv.serve_model("foo", test_mode=True)
    generate_func = app_instance.func

    request = mock.AsyncMock()
    request.json.return_value = {"prompt": "test"}

    result = await generate_func(request)
    assert result.content["sql"] == "SELECT * FROM maxtext_serve WHERE prompt='test'"
