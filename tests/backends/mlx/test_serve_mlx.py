"""Tests for MLX model serving without MagicMock."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import NoReturn
from unittest.mock import patch

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


def _noop_load_mlx(_name: str) -> None:
    """No-op MLX model preload hook."""
    return


def test_mlx_serve_model() -> None:
    """Verify mlx serve model behavior."""
    res = mlx_serve.serve_mlx_model("dummy_mlx_model", port=8080, max_batch_size=128, model_loader=_noop_load_mlx)
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
    captured_kwargs: dict[str, Callable[..., str | list[str] | None] | float] = {}

    def _capture_create(
        backend_name: str,
        model_name: str,
        **kw: Callable[..., str | list[str] | None] | float,
    ) -> str:
        """Capture `create_common_app` keyword arguments.

        Returns:
            Sentinel `'app'` string.
        """
        _ = (backend_name, model_name)
        captured_kwargs.update(kw)
        return "app"

    load_calls: list[str] = []

    monkeypatch.setattr("gemma_4_sql.backends.common_serve.create_common_app", _capture_create)
    assert (
        mlx_serve._app_factory(
            "m",
            model_loader=load_calls.append,
            query_generator=lambda _p, _m: "SELECT 1",
        )
        == "app"
    )
    startup_cb = captured_kwargs["startup_callback"]
    assert callable(startup_cb)
    startup_cb()
    assert load_calls == []

    gen_cb = captured_kwargs["generate_logic"]
    batch_cb = captured_kwargs["batch_generate_logic"]
    assert callable(gen_cb) and callable(batch_cb)
    assert gen_cb("p") == "SELECT 1"
    assert batch_cb(["p1", "p2"]) == ["SELECT 1", "SELECT 1"]


def test_load_mlx_model_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify load mlx model missing deps behavior."""
    mlx_serve.clear_mlx_model_cache()
    monkeypatch.setattr(mlx_serve, "mx", None)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    with pytest.raises(ImportError):
        _load_mlx_model("test")


def test_generate_query_error() -> None:
    """Verify generate query error behavior."""
    with pytest.raises(InferenceError):
        _generate_query("SELECT 1", model_name="dummy")


def test_batch_generate_queries() -> None:
    """Verify batch generate queries behavior."""

    def _gen_select_1(_prompt: str, _model: str) -> str:
        """Return SELECT 1."""
        return "SELECT 1"

    res = _batch_generate_queries(["query1", "query2"], query_generator=_gen_select_1)
    assert res == ["SELECT 1", "SELECT 1"]


def test_app_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _app_factory serves real HTTP /health and /generate requests through FastAPI TestClient."""
    import gemma_4_sql.backends.mlx.inference as mlx_inf
    from tests.utils.test_utils import make_test_client

    def _stub_generate_sql(**_kw: str | int) -> dict[str, str]:
        """Return deterministic SQL dictionary."""
        return {"sql": "SELECT 42", "status": "success"}

    monkeypatch.setattr(mlx_inf, "generate_sql", _stub_generate_sql)
    app = _app_factory("dummy_model", model_loader=_noop_load_mlx)
    assert app is not None
    with make_test_client(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["backend"] == "mlx"

        gen = client.post("/generate", json={"prompt": "Answer"})
        assert gen.status_code == 200
        assert gen.json()["sql"] == "SELECT 42"


def test_load_mlx_model_cache_and_branches() -> None:
    """Verify _load_mlx_model caching and eviction without MagicMock."""
    mlx_serve._mlx_model_cache.clear()
    load_ret: tuple[str, str] | str = ("model_obj", "tokenizer_obj")

    class _FakeMlxLmModule:
        """Typed `mlx_lm` module double."""

        @staticmethod
        def load(_name: str) -> tuple[str, str] | str:
            """Return `load_ret`."""
            return load_ret

    with patch.dict(sys.modules, {"mlx_lm": _FakeMlxLmModule}):
        m, t = mlx_serve._load_mlx_model("test_tuple")
        assert m == "model_obj"
        assert t == "tokenizer_obj"

        m2, _t2 = mlx_serve._load_mlx_model("test_tuple")
        assert m2 == "model_obj"

        load_ret = "model_only"
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
        """Provide mock load mlx model test helper.

        Raises:
            OSError: Always raised to simulate an OS error.
        """
        raise OSError("simulated OS error")

    monkeypatch.setattr(mlx_serve, "mx", "mock_mx")

    def mock_create_common_app(
        backend_name: str,
        model_name: str,
        startup_callback: Callable[[], None],
        **_kwargs: Callable[..., str | list[str]] | float,
    ) -> str:
        """Provide mock create common app test helper.

        Returns:
            Sentinel `'app'` string.
        """
        _ = (backend_name, model_name)
        startup_callback()
        return "app"

    monkeypatch.setattr("gemma_4_sql.backends.common_serve.create_common_app", mock_create_common_app)
    app = mlx_serve._app_factory("test_model", model_loader=mock_load_mlx_model)
    assert app == "app"


def test_generate_query_success_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _generate_query success and error handling without MagicMock."""
    import gemma_4_sql.backends.mlx.inference as mlx_inf

    mode = "ok"

    def _fake_gen_sql(**_kw: str | int | bool) -> dict[str, str]:
        """Return SQL result or raise based on `mode`.

        Returns:
            Generated SQL dictionary.

        Raises:
            ValueError: If `mode` is `'val_err'`.
            InferenceError: If `mode` is `'inf_err'`.
        """
        if mode == "val_err":
            raise ValueError("generic error")
        if mode == "inf_err":
            raise InferenceError("my inference error")
        if mode == "empty":
            return {"sql": ""}
        return {"sql": "SELECT 1"}

    monkeypatch.setattr(mlx_inf, "generate_sql", _fake_gen_sql)
    assert mlx_serve._generate_query("prompt") == "SELECT 1"

    mode = "empty"
    with pytest.raises(InferenceError, match="empty SQL"):
        mlx_serve._generate_query("prompt")

    mode = "val_err"
    with pytest.raises(InferenceError, match="MLX generation failed: generic error"):
        mlx_serve._generate_query("prompt")

    mode = "inf_err"
    with pytest.raises(InferenceError, match="my inference error"):
        mlx_serve._generate_query("prompt")
