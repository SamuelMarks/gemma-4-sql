"""Behavioral tests for the SDK Serve continuous batching module."""

from __future__ import annotations

import json

import pytest

import gemma_4_sql.backends.pytorch.serve as pt_serve
from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError
from gemma_4_sql.sdk.serve import serve_model, serve_model_handle
from tests.conftest import requires_jax
from tests.utils.test_utils import RecordingBackendDouble


@requires_jax
def test_serve_model_jax_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify serve_model_handle creates a FastAPI continuous batching app with JSON-serializable metadata."""
    from gemma_4_sql.backends.common_serve import ServeApplicationBundle
    from gemma_4_sql.type_hints import FastAPIAppProtocol
    from tests.utils.test_utils import make_test_client

    double = RecordingBackendDouble("jax")
    monkeypatch.setattr("gemma_4_sql.backends.jax.inference.generate_sql", double.generate_sql)
    handle = serve_model_handle("foo", port=8123, max_batch_size=16, backend="jax")
    res = handle.metadata
    assert json.loads(json.dumps(res))["model"] == "foo"
    assert res["backend"] == "jax"
    assert res["model"] == "foo"
    assert res["port"] == 8123
    assert res["max_batch_size"] == 16
    assert res.get("mode") == "continuous_batching"
    app = handle.app
    assert isinstance(app, ServeApplicationBundle)
    assert isinstance(app, FastAPIAppProtocol)

    with make_test_client(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        assert ready.json()["max_batch_size"] == 16

        models = client.get("/v1/models")
        assert models.status_code == 200
        models_data = models.json()["data"]
        assert isinstance(models_data, list)
        assert isinstance(models_data[0], dict)
        assert models_data[0]["id"] == "foo"

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
