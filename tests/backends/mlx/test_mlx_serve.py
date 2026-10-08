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
