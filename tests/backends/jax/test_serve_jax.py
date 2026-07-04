# Copyright 2024
"""Provide module docstring."""

import sys
from unittest import mock

import pytest

import gemma_4_sql.backends.jax.serve as srv


def test_serve_model_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function test_serve_model_jax.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(srv, "jax", object())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", mock.MagicMock())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.uvicorn", mock.MagicMock())
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["model"] == "foo":
        raise AssertionError
    if not res["port"] == int("8000"):
        raise AssertionError
    if not res["max_batch_size"] == int("16"):
        raise AssertionError
    if not res["mode"] == "continuous_batching":
        raise AssertionError
    if not res["status"] == "running_jax_serve":
        raise AssertionError


def test_serve_model_jax_missing() -> None:
    """Initialize function test_serve_model_jax_missing.

    Raises:
        AssertionError: Description.

    """
    with mock.patch.dict(sys.modules, {"jax": None}):
        importlib = __import__("importlib", fromlist=[""])
        importlib.reload(srv)
        res = srv.serve_model("foo")
        if not res["status"] == "mocked_missing_jax":
            raise AssertionError


def test_serve_model_jax_missing_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test missing FastAPI.

    Raises:
        AssertionError: Description.

    """
    importlib = __import__("importlib", fromlist=[""])
    importlib.reload(srv)
    monkeypatch.setattr(srv, "jax", object())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", None)
    res = srv.serve_model("foo")
    if not res["status"] == "failed_missing_fastapi":
        raise AssertionError


@pytest.mark.asyncio
async def test_generate_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate endpoint logic directly.

    Raises:
        AssertionError: Description.

    """
    importlib = __import__("importlib", fromlist=[""])
    importlib.reload(srv)
    monkeypatch.setattr(srv, "jax", object())
    mock_app = mock.MagicMock()
    mock_app.router.routes = []

    def mock_post(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """

        def decorator(func: object) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
            route = mock.MagicMock()
            route.endpoint = func
            mock_app.router.routes.append(route)
            return func

        return decorator

    mock_app.post = mock_post
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", lambda *_args, **_kwargs: mock_app)
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.uvicorn", mock.MagicMock())
    res = srv.serve_model("foo", test_mode=True)
    app = res["app"]
    generate_func = app.router.routes[-1].endpoint
    request = mock.AsyncMock()
    request.json.return_value = {"prompt": "test"}
    result = await generate_func(request)
    sql_val = result.body.decode() if hasattr(result, "body") else result["sql"]
    if "SELECT * FROM generated WHERE prompt='test'" not in sql_val:
        raise AssertionError


def test_serve_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    mdl = __import__("gemma_4_sql.backends.jax.serve", fromlist=[""])
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "fastapi", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    importlib.reload(mdl)
