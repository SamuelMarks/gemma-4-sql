"""Tests for MLX model serving."""

from __future__ import annotations

from collections.abc import Callable
from typing import NoReturn
from unittest.mock import MagicMock, patch

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
from tests.conftest import requires_mlx

pytestmark = requires_mlx


class _DummySentinel:
    """Sentinel stub for loaded MLX model and tokenizer."""


def test_mlx_serve_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify mlx serve model behavior."""
    monkeypatch.setattr(mlx_serve, "_load_mlx_model", lambda _name: (_DummySentinel(), _DummySentinel()))  # pyright: ignore[reportUnknownLambdaType]
    res = serve_model("dummy_mlx_model", port=8080, max_batch_size=128)
    assert res["status"] == "running_mlx_serve"
    assert res["backend"] == "mlx"
    assert res["model"] == "dummy_mlx_model"
    assert res["port"] == 8080
    assert res["max_batch_size"] == 128

    with patch.object(mlx_serve, "serve_model_wrapper", return_value={"status": "failed"}):
        res_fail = serve_model("dummy_mlx_model")
        assert res_fail["status"] == "failed"


def test_mlx_serve_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify mlx serve missing deps behavior."""
    monkeypatch.setattr(mlx_serve, "mx", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing for serve"):
        serve_model("model")


def test_app_factory_mx_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _app_factory startup callback is a no-op when mx is None."""
    monkeypatch.setattr(mlx_serve, "mx", None)
    with patch("gemma_4_sql.backends.common_serve.create_common_app") as mock_create:
        mock_create.return_value = "app"
        mlx_serve._app_factory("m")
        kwargs = mock_create.call_args[1]
        with patch.object(mlx_serve, "_load_mlx_model") as mock_load:
            kwargs["startup_callback"]()
            mock_load.assert_not_called()
        with patch.object(mlx_serve, "_generate_query", return_value="SELECT 1"):
            assert kwargs["generate_logic"]("p") == "SELECT 1"
            assert kwargs["batch_generate_logic"](["p1", "p2"]) == ["SELECT 1", "SELECT 1"]


def test_load_mlx_model_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify load mlx model missing deps behavior."""
    import sys

    mlx_serve.clear_mlx_model_cache()
    monkeypatch.setattr(mlx_serve, "mx", None)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    with pytest.raises(ImportError):
        _load_mlx_model("test")


def test_generate_query_error() -> None:
    """Verify generate query error behavior."""
    with pytest.raises(InferenceError):
        _generate_query("SELECT 1", model_name="dummy")


def test_batch_generate_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify batch generate queries behavior."""
    monkeypatch.setattr(mlx_serve, "_generate_query", lambda *_a, **_k: "SELECT 1")  # pyright: ignore[reportUnknownLambdaType]
    res = _batch_generate_queries(["query1", "query2"])
    assert res == ["SELECT 1", "SELECT 1"]


def test_app_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify app factory behavior."""
    monkeypatch.setattr(mlx_serve, "_load_mlx_model", lambda _name: (_DummySentinel(), _DummySentinel()))  # pyright: ignore[reportUnknownLambdaType]
    app = _app_factory("dummy_model")
    assert app is not None


def test_load_mlx_model_cache_and_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify load mlx model cache and branches behavior."""
    import sys

    mlx_serve._mlx_model_cache.clear()
    mock_mlxlm = MagicMock()
    mock_mlxlm.load.return_value = ("model_obj", "tokenizer_obj")
    monkeypatch.setitem(sys.modules, "mlx_lm", mock_mlxlm)

    m, t = mlx_serve._load_mlx_model("test_tuple")
    assert m == "model_obj"
    assert t == "tokenizer_obj"

    m2, _t2 = mlx_serve._load_mlx_model("test_tuple")
    assert m2 == "model_obj"

    mock_mlxlm.load.return_value = "model_only"
    m3, t3 = mlx_serve._load_mlx_model("test_single")
    assert m3 == "model_only"
    assert t3 is None

    for idx in range(mlx_serve.MAX_MLX_CACHE_SIZE + 2):
        mlx_serve._load_mlx_model(f"evict_{idx}")
    assert len(mlx_serve._mlx_model_cache) == mlx_serve.MAX_MLX_CACHE_SIZE
    mlx_serve.clear_mlx_model_cache()
    assert len(mlx_serve._mlx_model_cache) == 0


def test_app_factory_startup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify app factory startup error behavior."""

    def mock_load_mlx_model(_name: str) -> NoReturn:
        """Provide mock load mlx model test helper."""
        raise OSError("simulated OS error")

    monkeypatch.setattr(mlx_serve, "_load_mlx_model", mock_load_mlx_model)
    monkeypatch.setattr(mlx_serve, "mx", "mock_mx")

    def mock_create_common_app(
        backend_name: str,
        model_name: str,
        startup_callback: Callable[[], None],
        **_kwargs: Callable[..., str | list[str]] | float,
    ) -> str:
        """Provide mock create common app test helper."""
        _ = (backend_name, model_name)
        startup_callback()
        return "app"

    monkeypatch.setattr("gemma_4_sql.backends.common_serve.create_common_app", mock_create_common_app)
    app = mlx_serve._app_factory("test_model")
    assert app == "app"


def test_generate_query_success_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify generate query success and errors behavior."""
    import gemma_4_sql.backends.mlx.inference as mlx_inf

    mock_gen = MagicMock(return_value={"sql": "SELECT 1"})
    monkeypatch.setattr(mlx_inf, "generate_sql", mock_gen)
    assert mlx_serve._generate_query("prompt") == "SELECT 1"

    mock_gen.return_value = {"sql": ""}
    with pytest.raises(InferenceError, match="empty SQL"):
        mlx_serve._generate_query("prompt")

    mock_gen.side_effect = ValueError("generic error")
    with pytest.raises(InferenceError, match="MLX generation failed: generic error"):
        mlx_serve._generate_query("prompt")

    mock_gen.side_effect = InferenceError("my inference error")
    with pytest.raises(InferenceError, match="my inference error"):
        mlx_serve._generate_query("prompt")
