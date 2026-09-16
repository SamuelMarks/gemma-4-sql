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
            res = serve_model("foo", backend=backend, test_mode=True)
            assert res["backend"] == backend
        except DependencyMissingError:
            pass

    try:
        from unittest import mock

        monkeypatch.setattr("gemma_4_sql.backends.jax.inference.generate_sql", mock.MagicMock())
        res = serve_model("foo", backend="jax", test_mode=True)
        if not res["backend"] == "jax":
            raise AssertionError
    except DependencyMissingError:
        pass

    import gemma_4_sql.backends.pytorch.serve as pt_serve

    monkeypatch.setattr(pt_serve, "AsyncEngineArgs", None)
    with pytest.raises(DependencyMissingError):
        serve_model("foo", backend="pytorch")


def test_serve_model_routing_error() -> object:
    """Initialize function test_serve_model_routing_error."""
    with pytest.raises(ValueError, match=r".*"):
        serve_model("foo", backend="unknown")
