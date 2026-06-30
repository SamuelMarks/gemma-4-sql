"""Module docstring."""

import sys
from unittest import mock

import pytest

import gemma_4_sql.backends.jax.serve as srv


def test_serve_model_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function test_serve_model_jax."""
    monkeypatch.setattr(srv, "jax", object())
    monkeypatch.setattr(srv, "FastAPI", mock.MagicMock())
    monkeypatch.setattr(srv, "uvicorn", mock.MagicMock())

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
    """Initialize function test_serve_model_jax_missing."""
    with mock.patch.dict(sys.modules, {"jax": None}):
        importlib = __import__("importlib")
        importlib.reload(srv)
        res = srv.serve_model("foo")
        if not res["status"] == "mocked_missing_jax":
            raise AssertionError


def test_serve_model_jax_missing_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test missing FastAPI."""
    importlib = __import__("importlib")
    importlib.reload(srv)
    monkeypatch.setattr(srv, "jax", object())
    monkeypatch.setattr(srv, "FastAPI", None)
    res = srv.serve_model("foo")
    if not res["status"] == "failed_missing_fastapi":
        raise AssertionError


@pytest.mark.asyncio()
async def test_generate_endpoint() -> None:
    """Test generate endpoint logic directly."""
    importlib = __import__("importlib")
    importlib.reload(srv)
    srv.jax = object()

    res = srv.serve_model("foo", test_mode=True)
    app = res["app"]

    # Extract the function
    generate_func = app.router.routes[-1].endpoint

    request = mock.AsyncMock()
    request.json.return_value = {"prompt": "test"}
    result = await generate_func(request)
    if not result["sql"] == "SELECT * FROM generated WHERE prompt='test'":
        raise AssertionError
