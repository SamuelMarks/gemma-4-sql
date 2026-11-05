"""Tests for maxtext serve."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.maxtext.serve as serve_module
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


def test_maxtext_serve_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test maxtext serve missing deps."""
    monkeypatch.setattr(serve_module, "jax", None)
    monkeypatch.setattr(serve_module, "gemma4", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        serve_module.serve_model("model")


def test_maxtext_serve_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test maxtext serve success."""
    mock_jax = MagicMock()
    monkeypatch.setattr(serve_module, "jax", mock_jax)
    monkeypatch.setattr(serve_module, "gemma4", MagicMock())

    res = serve_module.serve_model("model", port=8090, max_batch_size=64)
    assert res["backend"] == "maxtext"
    assert res["port"] == 8090
    assert res["max_batch_size"] == 64
    assert res["status"] == "running_maxtext_serve"


def test_maxtext_create_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test maxtext _create_app with real FastAPI TestClient and direct execution."""
    from fastapi.testclient import TestClient

    from gemma_4_sql.backends.common_serve import DirectRunnableAppProtocol

    mock_jax = MagicMock()
    mock_jax.distributed.initialize.side_effect = [None, RuntimeError("already initialized")]
    monkeypatch.setattr(serve_module, "jax", mock_jax)
    monkeypatch.setattr(serve_module, "gemma4", MagicMock())

    app = serve_module._create_app("model")
    assert isinstance(app, DirectRunnableAppProtocol)
    app_error_startup = serve_module._create_app("model")
    assert app_error_startup is not None

    with patch("gemma_4_sql.backends.maxtext.inference.generate_sql") as mock_generate_sql:
        mock_generate_sql.return_value = {"sql": "SELECT 1", "status": "success"}
        assert app.run_direct("prompt") == "SELECT 1"

        with TestClient(app) as client:
            health_resp = client.get("/health")
            assert health_resp.status_code == 200
            assert health_resp.json()["backend"] == "maxtext"

            gen_resp = client.post("/generate", json={"prompt": "Count users"})
            assert gen_resp.status_code == 200
            assert gen_resp.json()["sql"] == "SELECT 1"

        mock_generate_sql.return_value = {"sql": "", "status": "failed: empty"}
        with pytest.raises(InferenceError, match="empty"):
            app.run_direct("prompt")

        mock_generate_sql.side_effect = RuntimeError("test error")
        with pytest.raises(InferenceError, match="MaxText generation failed"):
            app.run_direct("prompt")

        mock_generate_sql.side_effect = InferenceError("test error")
        with pytest.raises(InferenceError, match="test error"):
            app.run_direct("prompt")
