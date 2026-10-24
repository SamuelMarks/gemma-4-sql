"""Unit and behavioral tests for JAX continuous batching serve module."""

from __future__ import annotations

import sys
from typing import Any
from unittest import mock

import pytest

from tests.conftest import requires_jax

pytestmark = requires_jax

import gemma_4_sql.backends.jax.serve as srv
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


def test_serve_model_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `serve_model` configures startup warmup, single-prompt generation, and batch generation."""
    monkeypatch.setattr(srv, "jax", object())

    def mock_serve_model_wrapper(
        backend_name: str,
        model_name: str,
        port: int,
        max_batch_size: int,
        missing_deps: bool,
        app_factory: Any,
    ) -> dict[str, Any]:
        """Exercise the FastAPI app factory callbacks."""
        assert not missing_deps
        app = app_factory()

        startup = app.get("startup_callback")
        mock_module = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"gemma_4_sql.backends.jax.inference": mock_module}):
            startup()
            mock_module.generate_sql.assert_called_once_with(model_name=model_name, prompt="SELECT 1")

        # Startup warmup error should be logged and ignored
        mock_module.generate_sql.side_effect = RuntimeError("Warmup fail")
        with mock.patch.dict(sys.modules, {"gemma_4_sql.backends.jax.inference": mock_module}):
            startup()

        # Successful generation
        generate = app.get("generate_logic")
        mock_module.generate_sql.reset_mock(side_effect=True)
        mock_module.generate_sql.return_value = {"sql": "SELECT 42"}
        with mock.patch.dict(sys.modules, {"gemma_4_sql.backends.jax.inference": mock_module}):
            res = generate("test prompt")
            assert res == "SELECT 42"

        # Empty SQL raises InferenceError
        mock_module.generate_sql.return_value = {"other": "data"}
        with (
            mock.patch.dict(sys.modules, {"gemma_4_sql.backends.jax.inference": mock_module}),
            pytest.raises(InferenceError, match="JAX inference returned empty SQL"),
        ):
            generate("test prompt")

        # Underlying exception raises InferenceError
        mock_module.generate_sql.reset_mock(side_effect=True)
        mock_module.generate_sql.side_effect = RuntimeError("Gen fail")
        with (
            mock.patch.dict(sys.modules, {"gemma_4_sql.backends.jax.inference": mock_module}),
            pytest.raises(InferenceError, match="JAX generation failed"),
        ):
            generate("test prompt")

        # Batch generation
        batch_generate = app.get("batch_generate_logic")
        mock_module.generate_sql.reset_mock(side_effect=True)
        mock_module.generate_sql.return_value = {"sql": "SELECT 42"}
        with mock.patch.dict(sys.modules, {"gemma_4_sql.backends.jax.inference": mock_module}):
            res_batch = batch_generate(["prompt 1", "prompt 2"])
            assert res_batch == ["SELECT 42", "SELECT 42"]

        return {
            "backend": backend_name,
            "model": model_name,
            "port": port,
            "max_batch_size": max_batch_size,
            "mode": "continuous_batching",
            "status": "running_jax_serve",
        }

    def mock_create_common_app(**kwargs: Any) -> dict[str, Any]:
        """Return the keyword arguments passed to `create_common_app`."""
        return kwargs

    monkeypatch.setattr(srv, "serve_model_wrapper", mock_serve_model_wrapper)
    monkeypatch.setattr(srv, "create_common_app", mock_create_common_app)

    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    assert res["backend"] == "jax"
    assert res["model"] == "foo"
    assert res["port"] == 8000
    assert res["max_batch_size"] == 16
    assert res.get("mode") == "continuous_batching"
    assert res["status"] == "running_jax_serve"


def test_serve_model_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `serve_model` raises `DependencyMissingError` when JAX is unavailable."""
    monkeypatch.setattr(srv, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing for serve\."):
        srv.serve_model("foo")


def test_serve_model_jax_other_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `serve_model` returns wrapper status when server is not in running state."""
    monkeypatch.setattr(srv, "jax", object())

    def mock_serve_model_wrapper(**_kwargs: Any) -> dict[str, Any]:
        """Simulate a non-running status."""
        return {
            "backend": "jax",
            "model": "foo",
            "port": 8000,
            "max_batch_size": 256,
            "status": "backend_unavailable",
        }

    monkeypatch.setattr(srv, "serve_model_wrapper", mock_serve_model_wrapper)
    res = srv.serve_model("foo")
    assert res["status"] == "backend_unavailable"
