"""Provide module docstring."""

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.serve import serve_model


def test_serve_model_routing(monkeypatch: pytest.MonkeyPatch) -> object:
    """Initialize function test_serve_model_routing.

    Raises:
        AssertionError: Description.

    """
    for backend in ["keras", "maxtext"]:
        try:
            res = serve_model(
                "foo",
                backend=backend,
            )
            assert res["backend"] == backend
        except DependencyMissingError:
            pass

    from unittest import mock

    monkeypatch.setattr("gemma_4_sql.backends.jax.inference.generate_sql", mock.MagicMock())
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    jax_agent = get_backend("jax")
    monkeypatch.setattr(jax_agent, "serve_model", lambda *a, **k: {"backend": "jax"})

    res = serve_model(
        "foo",
        backend="jax",
    )
    assert res["backend"] == "jax"
    import gemma_4_sql.backends.pytorch.serve as pt_serve

    monkeypatch.setattr(pt_serve, "AsyncEngineArgs", None)
    with pytest.raises(DependencyMissingError):
        serve_model("foo", backend="pytorch")


def test_serve_model_routing_error() -> object:
    """Initialize function test_serve_model_routing_error."""
    with pytest.raises(ValueError, match=r".*"):
        serve_model("foo", backend="unknown")
