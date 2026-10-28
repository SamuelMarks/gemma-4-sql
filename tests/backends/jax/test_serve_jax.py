"""Unit and behavioral tests for JAX continuous batching serve module."""

from __future__ import annotations

import sys
from unittest import mock

import pytest

from tests.conftest import requires_jax

pytestmark = requires_jax

import gemma_4_sql.backends.jax.serve as srv
from gemma_4_sql.backends.common_serve import DirectRunnableAppProtocol
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


def test_serve_model_jax() -> None:
    """Verify `serve_model` configures startup warmup, single-prompt generation, and batch generation through real `serve_model_wrapper`."""
    from fastapi.testclient import TestClient

    mock_module = mock.MagicMock()
    mock_module.generate_sql.return_value = {"sql": "SELECT 42"}
    with mock.patch.dict(sys.modules, {"gemma_4_sql.backends.jax.inference": mock_module}):
        res = srv.serve_model("foo", port=8000, max_batch_size=16)
        assert res["backend"] == "jax"
        assert res["model"] == "foo"
        assert res["port"] == 8000
        assert res["max_batch_size"] == 16
        assert res.get("mode") == "continuous_batching"
        assert res["status"] == "running_jax_serve"
        app = res.get("app")
        assert isinstance(app, DirectRunnableAppProtocol)

        with TestClient(app) as client:
            resp = client.post("/generate", json={"prompt": "test prompt"})
            assert resp.status_code == 200
            assert resp.json()["sql"] == "SELECT 42"

        # Empty SQL raises InferenceError
        mock_module.generate_sql.return_value = {"other": "data"}
        with pytest.raises(InferenceError, match="JAX inference returned empty SQL"):
            app.run_direct("test prompt")

        # Underlying exception raises InferenceError
        mock_module.generate_sql.side_effect = RuntimeError("Gen fail")
        with pytest.raises(InferenceError, match="JAX generation failed"):
            app.run_direct("test prompt")

    # Startup warmup error should be logged and ignored during app initialization
    mock_fail_warmup = mock.MagicMock()
    mock_fail_warmup.generate_sql.side_effect = RuntimeError("Warmup fail")
    with mock.patch.dict(sys.modules, {"gemma_4_sql.backends.jax.inference": mock_fail_warmup}):
        res_warmup = srv.serve_model("foo", port=8000, max_batch_size=16)
        assert res_warmup["status"] == "running_jax_serve"


def test_serve_model_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `serve_model` raises `DependencyMissingError` when JAX is unavailable."""
    monkeypatch.setattr(srv, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing for serve\."):
        srv.serve_model("foo")


def test_serve_model_jax_other_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `serve_model` returns failed status when app creation raises an operational error."""
    monkeypatch.setattr(srv, "create_backend_serve_app", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("init fail")))  # pyright: ignore[reportUnknownLambdaType]
    res = srv.serve_model("foo")
    assert res["status"] == "failed: init fail"
