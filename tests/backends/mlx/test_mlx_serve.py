"""Tests for MLX model serving."""

from __future__ import annotations

import sys
from unittest import mock

import pytest

import gemma_4_sql.backends.mlx.serve as mlx_serve
from gemma_4_sql.backends.mlx.serve import (
    _app_factory,
    _batch_generate_queries,
    _generate_query,
    _load_mlx_model,
    serve_model,
)
from gemma_4_sql.exceptions import DependencyMissingError


def test_mlx_serve_model_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MLX serve_model execution in test_mode.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", mock.MagicMock())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.uvicorn", mock.MagicMock())
    res = serve_model("dummy_mlx_model", port=8080, max_batch_size=128, test_mode=True)
    assert res["status"] == "running_mlx_serve"
    assert res["backend"] == "mlx"
    assert res["model"] == "dummy_mlx_model"
    assert res["port"] == 8080
    assert res["max_batch_size"] == 128


def test_mlx_serve_model_with_mx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MLX serve_model with mx present and not test_mode.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setattr(mlx_serve, "mx", object())
    monkeypatch.setattr(mlx_serve, "_load_mlx_model", lambda _name: (object(), object()))
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", mock.MagicMock())
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.uvicorn", mock.MagicMock())
    res = serve_model("dummy_mlx_model", port=8080, max_batch_size=128, test_mode=False)
    assert res["status"] == "running_mlx_serve"


def test_mlx_serve_app_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MLX _app_factory generate logic.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setattr("gemma_4_sql.backends.common_serve.FastAPI", mock.MagicMock())
    app = _app_factory("dummy_model", test_mode=True)
    assert app is not None
    assert "SELECT 1" in _generate_query("SELECT 1", test_mode=True)


def test_mlx_serve_model_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MLX serve_model raises DependencyMissingError when MLX is missing.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setattr(mlx_serve, "mx", None)
    with pytest.raises(DependencyMissingError, match=r"MLX dependencies are missing for serve\."):
        serve_model("dummy_mlx_model", port=8080, test_mode=False)


def test_load_mlx_model_cached_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _load_mlx_model cache hit and error handling.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    mlx_serve._mlx_model_cache.clear()
    dummy_model = object()
    dummy_tokenizer = object()
    mock_load = mock.MagicMock(return_value=(dummy_model, dummy_tokenizer))
    monkeypatch.setitem(sys.modules, "mlx_lm", mock.MagicMock(load=mock_load))

    m1, t1 = _load_mlx_model("cached_model")
    assert m1 is dummy_model
    assert t1 is dummy_tokenizer
    assert mock_load.call_count == 1

    # Call again to verify cache hit
    m2, t2 = _load_mlx_model("cached_model")
    assert m2 is dummy_model
    assert t2 is dummy_tokenizer
    assert mock_load.call_count == 1

    # Test when load returns a non-tuple/list single model
    mock_load_single = mock.MagicMock(return_value=dummy_model)
    monkeypatch.setitem(sys.modules, "mlx_lm", mock.MagicMock(load=mock_load_single))
    m_single, t_single = _load_mlx_model("single_model")
    assert m_single is dummy_model
    assert t_single is None

    # Test error handling
    mock_fail = mock.MagicMock(side_effect=ValueError("Load failed"))
    monkeypatch.setattr("mlx_lm.load", mock_fail, raising=False)
    with pytest.raises(ValueError, match="Load failed"):
        _load_mlx_model("fail_model")


def test_generate_query_real_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _generate_query and _batch_generate_queries branches.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    # Success branch
    mock_gen = mock.MagicMock(return_value={"sql": "SELECT id FROM users"})
    monkeypatch.setattr("gemma_4_sql.backends.mlx.inference.generate_sql", mock_gen)
    sql = _generate_query("find users", test_mode=False, model_name="m")
    assert sql == "SELECT id FROM users"

    # Batch queries
    batch_res = _batch_generate_queries(["q1", "q2"], test_mode=False, model_name="m")
    assert batch_res == ["SELECT id FROM users", "SELECT id FROM users"]

    from gemma_4_sql.exceptions import InferenceError

    # Empty sql branch raises InferenceError
    mock_empty = mock.MagicMock(return_value={"sql": ""})
    monkeypatch.setattr("gemma_4_sql.backends.mlx.inference.generate_sql", mock_empty)
    with pytest.raises(InferenceError, match="empty SQL"):
        _generate_query("find empty", test_mode=False, model_name="m")

    # Exception branch raises InferenceError
    mock_err = mock.MagicMock(side_effect=RuntimeError("MLX error"))
    monkeypatch.setattr("gemma_4_sql.backends.mlx.inference.generate_sql", mock_err)
    with pytest.raises(InferenceError, match="MLX error"):
        _generate_query("find error", test_mode=False, model_name="m")


def test_app_factory_startup_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _app_factory startup callback execution.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    captured_callbacks = []

    def mock_create_common_app(**kwargs: object) -> object:
        """Capture create_common_app kwargs."""
        startup_cb = kwargs.get("startup_callback")
        if callable(startup_cb):
            captured_callbacks.append(startup_cb)
            startup_cb()
        return mock.MagicMock()

    monkeypatch.setattr("gemma_4_sql.backends.mlx.serve.create_common_app", mock_create_common_app)
    monkeypatch.setattr(mlx_serve, "mx", object())

    # Successful preload
    monkeypatch.setattr(mlx_serve, "_load_mlx_model", lambda _name: (object(), object()))
    _app_factory("test_model", test_mode=False)
    assert len(captured_callbacks) == 1

    # Exception inside preload
    def fail_preload(_name: str) -> None:
        """Raise preload exception."""
        msg = "Preload error"
        raise RuntimeError(msg)

    monkeypatch.setattr(mlx_serve, "_load_mlx_model", fail_preload)
    _app_factory("test_fail_model", test_mode=False)
    assert len(captured_callbacks) == 2

    # test_mode=True skips preload
    _app_factory("test_mode_model", test_mode=True)
    assert len(captured_callbacks) == 3
