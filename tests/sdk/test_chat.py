"""Module docstring."""

import pytest
from gemma_4_sql.sdk.chat import chat_turn


def test_chat_turn_routing() -> object:  # type: ignore[return]
    """Initialize function test_chat_turn_routing."""
    for backend in ["jax", "keras", "maxtext", "pytorch"]:
        res = chat_turn("foo", [], "prompt", backend=backend)
        if not res["backend"] == backend:
            raise AssertionError
        if not res["model"] == "foo":
            raise AssertionError
        if not len(res["history"]) == int("2"):  # type: ignore[arg-type]
            raise AssertionError


def test_chat_turn_routing_error() -> object:  # type: ignore[return]
    """Initialize function test_chat_turn_routing_error."""
    with pytest.raises(ValueError, match=r".*"):
        chat_turn("foo", [], "prompt", backend="unknown")
