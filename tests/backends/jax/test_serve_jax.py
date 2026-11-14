"""Unit and behavioral tests for JAX continuous batching serve module."""

from __future__ import annotations

from typing import NoReturn

import pytest

from tests.conftest import requires_jax

pytestmark = requires_jax

import gemma_4_sql.backends.jax.inference as jax_inf
import gemma_4_sql.backends.jax.serve as srv
from gemma_4_sql.backends.common_serve import DirectRunnableAppProtocol, get_serve_handle
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


def test_serve_model_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `serve_model` configures warmup, single generation, and batch generation via `serve_model_wrapper`."""
    from tests.utils.test_utils import make_test_client

    def _gen_select_42(*, model_name: str, prompt: str) -> dict[str, str]:
        """Return deterministic `SELECT 42` payload."""
        _ = (model_name, prompt)
        return {"sql": "SELECT 42"}

    def _gen_empty(*, model_name: str, prompt: str) -> dict[str, str]:
        """Return dictionary missing the `sql` key."""
        _ = (model_name, prompt)
        return {"other": "data"}

    def _gen_fail(*, model_name: str, prompt: str) -> NoReturn:
        """Raise RuntimeError during SQL generation.

        Raises:
            RuntimeError: Always raised to simulate generation failure.

        """
        _ = (model_name, prompt)
        raise RuntimeError("Gen fail")

    monkeypatch.setattr(jax_inf, "generate_sql", _gen_select_42)
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    assert res["backend"] == "jax"
    assert res["model"] == "foo"
    assert res["port"] == 8000
    assert res["max_batch_size"] == 16
    assert res.get("mode") == "continuous_batching"
    assert res["status"] == "running_jax_serve"
    app = get_serve_handle(res).app
    assert isinstance(app, DirectRunnableAppProtocol)

    with make_test_client(app) as client:
        resp = client.post("/generate", json={"prompt": "test prompt"})
        assert resp.status_code == 200
        assert resp.json()["sql"] == "SELECT 42"

    # Empty SQL raises InferenceError
    monkeypatch.setattr(jax_inf, "generate_sql", _gen_empty)
    with pytest.raises(InferenceError, match="JAX inference returned empty SQL"):
        app.run_direct("test prompt")

    # Underlying exception raises InferenceError
    monkeypatch.setattr(jax_inf, "generate_sql", _gen_fail)
    with pytest.raises(InferenceError, match="JAX generation failed"):
        app.run_direct("test prompt")

    # Startup warmup error should be logged and ignored during app initialization
    res_warmup = srv.serve_model("foo", port=8000, max_batch_size=16)
    assert res_warmup["status"] == "running_jax_serve"


def test_serve_model_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `serve_model` raises `DependencyMissingError` when JAX is unavailable."""
    monkeypatch.setattr(srv, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing for serve\."):
        srv.serve_model("foo")


def test_serve_model_jax_other_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `serve_model` returns failed status when app creation raises an operational error."""
    from gemma_4_sql.type_hints import FastAPIAppProtocol, JSONValue

    def _raise_init_fail(**_kwargs: JSONValue) -> FastAPIAppProtocol:
        """Raise RuntimeError.

        Raises:
            RuntimeError: Always raised to simulate FastAPI app creation failure.

        """
        raise RuntimeError("init fail")

    monkeypatch.setattr(srv, "create_backend_serve_app", _raise_init_fail)
    res = srv.serve_model("foo")
    assert res["status"] == "failed: init fail"
