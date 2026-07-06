"""Provide module docstring."""

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


def test_serve_model_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function test_serve_model_jax_missing.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(srv, "jax", None)
    with pytest.raises(DependencyMissingError, match="JAX dependencies are missing for serve."):
        srv.serve_model("foo")


def test_serve_model_jax_missing_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test missing FastAPI.

    Raises:
        AssertionError: Description.

    """
    __import__("importlib", fromlist=[""])

    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(srv, "jax", object())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", None)
    with pytest.raises(DependencyMissingError):
        srv.serve_model("foo")


@pytest.mark.asyncio
async def test_generate_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate endpoint logic directly."""
    __import__("importlib", fromlist=[""])
    monkeypatch.setattr(srv, "jax", object())

    class MockJSONResponse:
        """Mock class."""

        def __init__(self, content: dict) -> None:
            """Init."""
            self.content = content
            self.body = str(content).encode()

    class MockApp:
        """Mock app."""

        def __init__(self) -> None:
            """Init."""
            self.router = mock.MagicMock()
            self.router.routes = []

        def post(self, *_args: object, **_kwargs: object) -> object:
            """Post."""

            def decorator(func: object) -> object:
                """Decorator."""
                route = mock.MagicMock()
                route.endpoint = func
                self.router.routes.append(route)
                return func

            return decorator

    app_instance = MockApp()

    monkeypatch.setattr("gemma_4_sql.backends.common_serve.JSONResponse", MockJSONResponse)
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", lambda *_args, **_kwargs: app_instance)
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.uvicorn", mock.MagicMock())

    import gemma_4_sql.backends.common_serve

    gemma_4_sql.backends.common_serve.Request = mock.MagicMock()
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.Request", mock.MagicMock())

    srv.serve_model("foo", test_mode=True)
    generate_func = app_instance.router.routes[-1].endpoint

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
