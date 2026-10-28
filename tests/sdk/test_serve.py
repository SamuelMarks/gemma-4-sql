"""Behavioral tests for the SDK Serve continuous batching module."""

from __future__ import annotations

import pytest

import gemma_4_sql.backends.pytorch.serve as pt_serve
from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError
from gemma_4_sql.sdk.serve import serve_model
from tests.conftest import requires_jax
from tests.utils.test_utils import RecordingBackendDouble


@requires_jax
def test_serve_model_jax_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify serve_model initializes a working FastAPI continuous batching app for JAX."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    double = RecordingBackendDouble("jax")
    monkeypatch.setattr("gemma_4_sql.backends.jax.inference.generate_sql", double.generate_sql)
    res = serve_model("foo", port=8123, max_batch_size=16, backend="jax")
    assert res["backend"] == "jax"
    assert res["model"] == "foo"
    assert res["port"] == 8123
    assert res["max_batch_size"] == 16
    assert res.get("mode") == "continuous_batching"
    app = res.get("app")
    assert isinstance(app, FastAPI)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        assert ready.json()["max_batch_size"] == 16

        models = client.get("/v1/models")
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == "foo"

        gen = client.post("/generate", json={"prompt": "Count rows"})
        assert gen.status_code == 200
        assert gen.json()["sql"] == "SELECT * FROM jax_table;"
        assert len(double.generate_calls) >= 1
        assert double.generate_calls[-1][1] == "Count rows"


def test_serve_model_missing_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify serve_model raises DependencyMissingError when vLLM is unavailable."""
    monkeypatch.setattr(pt_serve, "AsyncEngineArgs", None)
    with pytest.raises(DependencyMissingError):
        serve_model("foo", backend="pytorch")


def test_serve_model_routing_error() -> None:
    """Verify serve_model raises ConfigurationError on an unknown backend."""
    with pytest.raises(ConfigurationError, match="Unknown backend: unknown"):
        serve_model("foo", backend="unknown")
