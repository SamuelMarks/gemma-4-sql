"""Tests for MLX model serving."""

from __future__ import annotations

import pytest

import gemma_4_sql.backends.mlx.serve as mlx_serve
from gemma_4_sql.backends.mlx.serve import (
    _app_factory,
    _batch_generate_queries,
    _generate_query,
    _load_mlx_model,
    serve_model,
)
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


def test_mlx_serve_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docstring for test_mlx_serve_model."""
    monkeypatch.setattr(mlx_serve, "_load_mlx_model", lambda name: (object(), object()))
    res = serve_model("dummy_mlx_model", port=8080, max_batch_size=128)
    assert res["status"] == "running_mlx_serve"
    assert res["backend"] == "mlx"
    assert res["model"] == "dummy_mlx_model"
    assert res["port"] == 8080
    assert res["max_batch_size"] == 128


def test_mlx_serve_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docstring for test_mlx_serve_missing_deps."""
    monkeypatch.setattr(mlx_serve, "mx", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing for serve"):
        serve_model("model")


def test_load_mlx_model_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docstring for test_load_mlx_model_missing_deps."""
    monkeypatch.setattr(mlx_serve, "mx", None)
    with pytest.raises(ImportError):
        # We simulate what happens if mlx_lm is missing
        # mlx_lm is required for load
        # Let's mock mlx_lm load
        import builtins

        real_import = builtins.__import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Docstring for mock_import."""
            if name == "mlx_lm":
                raise ImportError("No module named mlx_lm")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        _load_mlx_model("test")


def test_generate_query_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docstring for test_generate_query_error."""
    with pytest.raises(InferenceError):
        _generate_query("SELECT 1", model_name="dummy")


def test_batch_generate_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docstring for test_batch_generate_queries."""

    def dummy_generate(*args, **kwargs):
        """Docstring for dummy_generate."""
        return "SELECT 1"

    monkeypatch.setattr(mlx_serve, "_generate_query", dummy_generate)
    res = _batch_generate_queries(["query1", "query2"])
    assert res == ["SELECT 1", "SELECT 1"]


def test_app_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docstring for test_app_factory."""
    monkeypatch.setattr(mlx_serve, "_load_mlx_model", lambda name: (object(), object()))
    app = _app_factory("dummy_model")
    assert app is not None


def test_load_mlx_model_cache_and_branches(monkeypatch):
    """Docstring for test_load_mlx_model_cache_and_branches."""
    import builtins
    from unittest.mock import MagicMock

    import gemma_4_sql.backends.mlx.serve as mlx_serve

    # clear cache
    mlx_serve._mlx_model_cache.clear()

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "mlx_lm":
            mock_mlxlm = MagicMock()
            # return tuple
            mock_mlxlm.load.return_value = ("model_obj", "tokenizer_obj")
            return mock_mlxlm
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Load (returns tuple)
    m, t = mlx_serve._load_mlx_model("test_tuple")
    assert m == "model_obj"
    assert t == "tokenizer_obj"

    # Load again (cache hit, line 40)
    m2, t2 = mlx_serve._load_mlx_model("test_tuple")
    assert m2 == "model_obj"

    def mock_import_single(name, *args, **kwargs):
        """Docstring for mock_import_single."""
        if name == "mlx_lm":
            mock_mlxlm = MagicMock()
            # return single object
            mock_mlxlm.load.return_value = "model_only"
            return mock_mlxlm
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import_single)
    m3, t3 = mlx_serve._load_mlx_model("test_single")
    assert m3 == "model_only"
    assert t3 is None


def test_app_factory_startup_error(monkeypatch):
    """Docstring for test_app_factory_startup_error."""
    import gemma_4_sql.backends.mlx.serve as mlx_serve

    def mock_load_mlx_model(name):
        """Docstring for mock_load_mlx_model."""
        raise OSError("simulated OS error")

    monkeypatch.setattr(mlx_serve, "_load_mlx_model", mock_load_mlx_model)
    monkeypatch.setattr(mlx_serve, "mx", "mock_mx")

    def mock_create_common_app(backend_name, model_name, startup_callback, **kwargs):
        """Docstring for mock_create_common_app."""
        startup_callback()  # invoke it!
        return "app"

    monkeypatch.setattr(mlx_serve, "create_common_app", mock_create_common_app)

    # Should not raise, just log warning
    app = mlx_serve._app_factory("test_model")
    assert app == "app"


def test_generate_query_success_and_errors(monkeypatch):
    """Docstring for test_generate_query_success_and_errors."""
    import builtins
    from unittest.mock import MagicMock

    import pytest

    import gemma_4_sql.backends.mlx.serve as mlx_serve
    from gemma_4_sql.exceptions import InferenceError

    mock_inf = MagicMock()

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "gemma_4_sql.backends.mlx.inference":
            return mock_inf
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # success
    mock_inf.backends.mlx.inference.generate_sql.return_value = {"sql": "SELECT 1"}
    assert mlx_serve._generate_query("prompt") == "SELECT 1"

    # empty
    mock_inf.backends.mlx.inference.generate_sql.return_value = {"sql": ""}
    with pytest.raises(InferenceError, match="empty SQL"):
        mlx_serve._generate_query("prompt")

    # generic error
    mock_inf.backends.mlx.inference.generate_sql.side_effect = ValueError("generic error")
    with pytest.raises(InferenceError, match="MLX generation failed: generic error"):
        mlx_serve._generate_query("prompt")

    # inference error
    mock_inf.backends.mlx.inference.generate_sql.side_effect = InferenceError("my inference error")
    with pytest.raises(InferenceError, match="my inference error"):
        mlx_serve._generate_query("prompt")
