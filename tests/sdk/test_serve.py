"""Module docstring."""

import pytest

from gemma_4_sql.sdk.serve import serve_model


def test_serve_model_routing() -> object:  # type: ignore[return]
    """Initialize function test_serve_model_routing."""
    for backend in ["jax", "keras", "maxtext", "pytorch"]:
        res = serve_model("foo", backend=backend)
        if not res["backend"] == backend:
            raise AssertionError
        if not res["model"] == "foo":
            raise AssertionError


def test_serve_model_routing_error() -> object:  # type: ignore[return]
    """Initialize function test_serve_model_routing_error."""
    with pytest.raises(ValueError, match=r".*"):
        serve_model("foo", backend="unknown")
