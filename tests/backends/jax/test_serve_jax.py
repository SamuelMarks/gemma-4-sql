"""Module docstring."""

import sys
from unittest import mock

import gemma_4_sql.backends.jax.serve as srv


def test_serve_model_jax() -> object:  # type: ignore[return]
    """Initialize function test_serve_model_jax."""
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["model"] == "foo":
        raise AssertionError
    if not res["port"] == int("8000"):
        raise AssertionError
    if not res["max_batch_size"] == int("16"):
        raise AssertionError
    if not res["mode"] == "continuous_batching":
        raise AssertionError


def test_serve_model_jax_missing() -> object:  # type: ignore[return]
    """Initialize function test_serve_model_jax_missing."""
    with mock.patch.dict(sys.modules, {"jax": None}):
        importlib = __import__("importlib")
        importlib.reload(srv)
        res = srv.serve_model("foo")
        if not res["status"] == "mocked_missing_jax":
            raise AssertionError
