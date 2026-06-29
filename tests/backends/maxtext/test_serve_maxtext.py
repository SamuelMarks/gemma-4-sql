"""Module docstring."""

import sys
from unittest import mock

import gemma_4_sql.backends.maxtext.serve as srv


def test_serve_model_maxtext() -> object:  # type: ignore[return]
    """Initialize function test_serve_model_maxtext."""
    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["model"] == "foo":
        raise AssertionError
    if not res["port"] == int("8000"):
        raise AssertionError
    if not res["max_batch_size"] == int("16"):
        raise AssertionError
    if not res["mode"] == "continuous_batching":
        raise AssertionError


def test_serve_model_maxtext_missing() -> object:  # type: ignore[return]
    """Initialize function test_serve_model_maxtext_missing."""
    with mock.patch.dict(sys.modules, {"maxtext.models.gemma4": None, "maxtext.models": None, "maxtext": None}):
        importlib = __import__("importlib")
        importlib.reload(srv)
        res = srv.serve_model("foo")
        if not res["status"] == "mocked_missing_maxtext":
            raise AssertionError
