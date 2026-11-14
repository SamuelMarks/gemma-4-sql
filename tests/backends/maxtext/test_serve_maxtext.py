"""Tests for MaxText continuous batching serve pipeline without MagicMock."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import gemma_4_sql.backends.maxtext.serve as serve_module
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


class _FakeDistributed:
    """Typed `jax.distributed` double."""

    def __init__(self, *, fail: bool = False) -> None:
        """Store failure toggle."""
        self.fail = fail

    def initialize(self) -> None:
        """No-op or raise `RuntimeError`.

        Raises:
            RuntimeError: If `self.fail` is True.

        """
        if self.fail:
            raise RuntimeError("already initialized")


class _FakeJax:
    """Typed `jax` double with `.distributed`."""

    def __init__(self, *, fail: bool = False) -> None:
        """Initialize `.distributed`."""
        self.distributed = _FakeDistributed(fail=fail)


class _FakeGemma4Module:
    """Sentinel `maxtext.models.gemma4` module substitute."""


def test_maxtext_serve_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test `serve_model` raises `DependencyMissingError` when MaxText dependencies are missing."""
    monkeypatch.setattr(serve_module, "jax", None)
    monkeypatch.setattr(serve_module, "gemma4", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        serve_module.serve_model("model")


def test_maxtext_serve_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test `serve_model` returns running `ServeResultDict` when dependencies are present."""
    monkeypatch.setattr(serve_module, "jax", _FakeJax())
    monkeypatch.setattr(serve_module, "gemma4", _FakeGemma4Module())

    res = serve_module.serve_model("model", port=8090, max_batch_size=64)
    assert res["backend"] == "maxtext"
    assert res["port"] == 8090
    assert res["max_batch_size"] == 64
    assert res["status"] == "running_maxtext_serve"


def test_maxtext_create_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test `_create_app` with real FastAPI `TestClient` and direct execution."""
    from gemma_4_sql.backends.common_serve import DirectRunnableAppProtocol
    from tests.utils.test_utils import make_test_client

    monkeypatch.setattr(serve_module, "jax", _FakeJax(fail=False))
    monkeypatch.setattr(serve_module, "gemma4", _FakeGemma4Module())

    app = serve_module._create_app("model")
    assert isinstance(app, DirectRunnableAppProtocol)

    monkeypatch.setattr(serve_module, "jax", _FakeJax(fail=True))
    app_error_startup = serve_module._create_app("model")
    assert app_error_startup is not None

    with patch("gemma_4_sql.backends.maxtext.inference.generate_sql") as mock_generate_sql:
        mock_generate_sql.return_value = {"sql": "SELECT 1", "status": "success"}
        assert app.run_direct("prompt") == "SELECT 1"

        with make_test_client(app) as client, make_test_client(app_error_startup) as client2:
            health_resp = client.get("/health")
            assert health_resp.status_code == 200
            assert health_resp.json()["backend"] == "maxtext"
            assert client2.get("/health").status_code == 200

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
