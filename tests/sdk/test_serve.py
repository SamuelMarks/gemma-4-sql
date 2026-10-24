"""Behavioral tests for the SDK Serve continuous batching module."""

from __future__ import annotations

import pytest

import gemma_4_sql.backends.pytorch.serve as pt_serve
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.serve import serve_model


def test_serve_model_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify serve_model initializes FastAPI continuous batching apps across backends."""
    monkeypatch.setattr("gemma_4_sql.backends.jax.inference.generate_sql", lambda **_k: {"sql": "SELECT 1"})
    for backend in ["keras", "maxtext", "jax"]:
        try:
            res = serve_model("foo", port=8123, max_batch_size=16, backend=backend)
            assert res["backend"] == backend
            assert res["model"] == "foo"
            assert res["port"] == 8123
            assert res["max_batch_size"] == 16
            assert res.get("mode") == "continuous_batching"
            assert res.get("app") is not None
        except DependencyMissingError:
            pass

    monkeypatch.setattr(pt_serve, "AsyncEngineArgs", None)
    with pytest.raises(DependencyMissingError):
        serve_model("foo", backend="pytorch")


def test_serve_model_routing_error() -> None:
    """Verify serve_model raises ValueError on an unknown backend."""
    with pytest.raises(ValueError, match="Unknown backend: unknown"):
        serve_model("foo", backend="unknown")
