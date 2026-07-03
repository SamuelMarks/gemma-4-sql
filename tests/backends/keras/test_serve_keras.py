"""Tests for Keras Serve."""

from __future__ import annotations

from unittest import mock

import pytest

import gemma_4_sql.backends.keras.serve as srv


def test_serve_model_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(srv, "keras", None)
    "Execute function."
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if res["status"] != "mocked_missing_keras":
        raise AssertionError


def test_serve_model_keras_fastapi_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(srv, "tf", object())
    monkeypatch.setattr(srv, "FastAPI", None)
    monkeypatch.setattr(srv, "keras", object())
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if res["status"] != "failed_missing_fastapi":
        raise AssertionError


def test_serve_model_keras_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(srv, "tf", object())
    monkeypatch.setattr(srv, "FastAPI", None)
    monkeypatch.setattr(srv, "keras", object())
    monkeypatch.setattr(srv, "FastAPI", object())
    monkeypatch.setattr(srv, "uvicorn", mock.MagicMock())

    def mock_create_app(*_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return "app"

    monkeypatch.setattr(srv, "_create_app", mock_create_app)
    res = srv.serve_model("foo", port=8000, max_batch_size=16, test_mode=False)
    if res["status"] != "running_keras_serve":
        raise AssertionError


def test_serve_model_keras_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(srv, "tf", object())
    monkeypatch.setattr(srv, "FastAPI", None)
    monkeypatch.setattr(srv, "keras", object())
    monkeypatch.setattr(srv, "FastAPI", object())
    monkeypatch.setattr(srv, "uvicorn", mock.MagicMock())

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function."""
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(srv, "_create_app", raise_err)
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if "failed" not in str(res["status"]):
        raise AssertionError


@pytest.mark.asyncio
async def test_generate_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate endpoint logic directly."""
    builtins = __import__("builtins")
    orig_import = builtins.__import__

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

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Execute function."""
        if name == "fastapi":
            return type("FastAPIMod", (), {"FastAPI": lambda *_args, **_kwargs: app_instance})
        if name == "fastapi.responses":

            class MockJSONResponse:
                """Provide class docstring."""

                def __init__(self, content: object) -> None:
                    """Execute function."""
                    self.content = content

            return type("ResponsesMod", (), {"JSONResponse": MockJSONResponse})
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)
    srv.create_app("foo", test_mode=True)
    generate_func = app_instance.func
    request = mock.AsyncMock()
    request.json.return_value = {"prompt": "test"}
    res = await generate_func(request)
    if res.content != {"sql": "SELECT * FROM keras_serve WHERE prompt='test'"}:
        raise AssertionError
