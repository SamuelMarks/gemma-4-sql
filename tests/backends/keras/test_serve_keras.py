"""Tests for Keras Serve."""

from __future__ import annotations

from unittest import mock

import pytest

import gemma_4_sql.backends.keras.serve as srv


def test_serve_model_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(srv, "keras", None)
    with pytest.raises(DependencyMissingError, match=r"Keras dependencies are missing for serve\."):
        srv.serve_model("foo", port=8000, max_batch_size=16)


def test_serve_model_keras_fastapi_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(srv, "tf", object())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", None)
    monkeypatch.setattr(srv, "keras", object())
    with pytest.raises(DependencyMissingError):
        srv.serve_model("foo", port=8000, max_batch_size=16)


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


@pytest.mark.asyncio
async def test_keras_serve_generation_logic_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _generate and _batch_generate branches in Keras serve.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    captured_kwargs: dict[str, object] = {}

    def mock_create_common_app(**kwargs: object) -> object:
        """Capture create_common_app kwargs."""
        captured_kwargs.update(kwargs)
        startup_cb = kwargs.get("startup_callback")
        if callable(startup_cb):
            startup_cb()
        return mock.MagicMock()

    monkeypatch.setattr("gemma_4_sql.backends.keras.serve.create_common_app", mock_create_common_app)

    # 1. Startup preload succeeds and loaded_model.generate works
    mock_model = mock.MagicMock()
    mock_model.generate.return_value = "SELECT name FROM students"
    mock_gemma_cls = mock.MagicMock()
    mock_gemma_cls.from_preset.return_value = mock_model
    mock_keras_nlp = mock.MagicMock(GemmaCausalLM=mock_gemma_cls)
    monkeypatch.setitem(__import__("sys").modules, "keras_nlp", mock.MagicMock())
    monkeypatch.setitem(__import__("sys").modules, "keras_nlp.models", mock_keras_nlp)

    srv.create_app("test_preset_model", test_mode=False)
    startup_cb = captured_kwargs["startup_callback"]
    assert callable(startup_cb)
    startup_cb()

    gen_cb = captured_kwargs["generate_logic"]
    batch_cb = captured_kwargs["batch_generate_logic"]
    assert callable(gen_cb)
    assert callable(batch_cb)

    assert gen_cb("query") == "SELECT name FROM students"

    # Batch generate with loaded_model
    mock_model.generate.return_value = ["SELECT 1", "SELECT 2"]
    assert batch_cb(["q1", "q2"]) == ["SELECT 1", "SELECT 2"]

    # Batch generate with error falls back to individual generate
    mock_model.generate.side_effect = RuntimeError("batch fail")
    monkeypatch.setattr("gemma_4_sql.backends.keras.inference.generate_sql", lambda **_kw: {"sql": "SELECT fallback"})
    assert batch_cb(["q1"]) == ["SELECT fallback"]

    # Generate with generate_sql fallback
    monkeypatch.setattr("gemma_4_sql.backends.keras.inference.generate_sql", lambda **_kw: {"sql": "SELECT 42"})
    mock_model.generate.side_effect = RuntimeError("single fail")
    assert gen_cb("query") == "SELECT 42"

    # Generate with all failing raises InferenceError
    from gemma_4_sql.exceptions import InferenceError

    monkeypatch.setattr("gemma_4_sql.backends.keras.inference.generate_sql", mock.MagicMock(side_effect=RuntimeError("all fail")))
    with pytest.raises(InferenceError, match="all fail"):
        gen_cb("query")

    # Generate with empty SQL raises and re-raises InferenceError
    monkeypatch.setattr("gemma_4_sql.backends.keras.inference.generate_sql", lambda **_kw: {"sql": ""})
    with pytest.raises(InferenceError, match="returned empty SQL"):
        gen_cb("query")

    # Startup preload fails gracefully
    mock_gemma_cls.from_preset.side_effect = ValueError("Preset failed")
    srv.create_app("fail_model", test_mode=False)
    captured_kwargs["startup_callback"]()

    # Startup with test_mode=True
    srv.create_app("test_mode_model", test_mode=True)
    captured_kwargs["startup_callback"]()
    gen_tm = captured_kwargs["generate_logic"]
    batch_tm = captured_kwargs["batch_generate_logic"]
    assert "keras_serve" in gen_tm("test_prompt")
    assert "keras_serve" in batch_tm(["test_prompt"])[0]

    # Test loaded_model is None execution of _generate and _batch_generate
    mock_gemma_cls.from_preset.return_value = None
    srv.create_app("no_model", test_mode=False)
    gen_none = captured_kwargs["generate_logic"]
    batch_none = captured_kwargs["batch_generate_logic"]
    monkeypatch.setattr("gemma_4_sql.backends.keras.inference.generate_sql", lambda **_kw: {"sql": "SELECT nonemodel"})
    assert gen_none("q") == "SELECT nonemodel"
    assert batch_none(["q"]) == ["SELECT nonemodel"]
