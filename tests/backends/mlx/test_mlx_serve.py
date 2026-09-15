"""Tests for MLX model serving."""

from __future__ import annotations

from unittest import mock

import pytest

import gemma_4_sql.backends.mlx.serve as mlx_serve
from gemma_4_sql.backends.mlx.serve import serve_model
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
    from gemma_4_sql.backends.mlx.serve import _app_factory, _generate_query

    app = _app_factory("dummy_model")
    assert app is not None
    assert "SELECT 1" in _generate_query("SELECT 1")


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
