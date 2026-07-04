# Copyright 2024
"""Tests for Keras Serve."""

from __future__ import annotations

from unittest import mock

import pytest

import gemma_4_sql.backends.keras.serve as srv


def test_serve_model_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(srv, "keras", None)
    "Execute function."
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if res["status"] != "mocked_missing_keras":
        raise AssertionError


def test_serve_model_keras_fastapi_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(srv, "tf", object())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", None)
    monkeypatch.setattr(srv, "keras", object())
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if res["status"] != "failed_missing_fastapi":
        raise AssertionError


def test_serve_model_keras_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(srv, "tf", object())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", None)
    monkeypatch.setattr(srv, "keras", object())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", object())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.uvicorn", mock.MagicMock())

    def mock_create_app(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "app"

    monkeypatch.setattr(srv, "create_app", mock_create_app)
    res = srv.serve_model("foo", port=8000, max_batch_size=16, test_mode=False)
    if res["status"] != "running_keras_serve":
        raise AssertionError


def test_serve_model_keras_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(srv, "tf", object())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", None)
    monkeypatch.setattr(srv, "keras", object())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", object())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.uvicorn", mock.MagicMock())

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(srv, "create_app", raise_err)
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if "failed" not in str(res["status"]):
        raise AssertionError


@pytest.mark.asyncio
async def test_generate_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate endpoint logic directly.

    Raises:
        AssertionError: Description.

    """
    builtins = __import__("builtins", fromlist=[""])
    orig_import = builtins.__import__

    mock_app = mock.MagicMock()
    mock_app.router.routes = []

    def mock_post(*_args: object, **_kwargs: object) -> object:
        """Docstring."""

        def decorator(func: object) -> object:
            """Docstring."""
            route = mock.MagicMock()
            route.endpoint = func
            mock_app.router.routes.append(route)
            return func

        return decorator

    mock_app.post = mock_post

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Docstring."""
        if name == "fastapi":
            return type("FastAPIMod", (), {"FastAPI": lambda *_args, **_kwargs: mock_app})
        if name == "fastapi.responses":

            class MockJSONResponse:
                """Docstring."""

                def __init__(self, content: object) -> None:
                    """Docstring."""
                    self.body = str(content).encode()

            return type("ResponsesMod", (), {"JSONResponse": MockJSONResponse})
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", lambda *_args, **_kwargs: mock_app)
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.JSONResponse", lambda content: type("MockJSONResponse", (), {"body": str(content).encode()}))

    srv.create_app("foo", test_mode=True)
    generate_func = mock_app.router.routes[-1].endpoint
    request = mock.AsyncMock()
    request.json.return_value = {"prompt": "test"}
    result = await generate_func(request)
    sql_val = result.body.decode() if hasattr(result, "body") else ""
    if "SELECT * FROM keras_serve WHERE prompt='test'" not in sql_val:
        raise AssertionError
