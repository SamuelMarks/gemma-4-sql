"""Tests for SDK Chat module."""

from typing import NoReturn as Never

import pytest

from gemma_4_sql.sdk.chat import chat_turn


def test_chat_turn_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test chat_turn with multiple backends.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    for backend in ["jax", "keras", "maxtext", "pytorch"]:
        backend_impl = get_backend(backend)
        monkeypatch.setattr(backend_impl, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
        res = chat_turn("foo", [{"role": "user", "content": "hi"}], "prompt", backend=backend)
        if res["backend"] != backend:
            raise AssertionError
        if res["model"] != "foo":
            raise AssertionError
        if res["response"] != "SELECT 1":
            raise AssertionError
        if len(res["history"]) != int("3"):
            raise AssertionError


def test_chat_turn_routing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test chat_turn error handling.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    backend_impl = get_backend("jax")

    def mock_generate(*_args: object, **_kwargs: object) -> Never:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "mock error"
        raise ValueError(msg)

    monkeypatch.setattr(backend_impl, "generate_sql", mock_generate)
    with pytest.raises(RuntimeError, match="Chat turn failed"):
        chat_turn("foo", [], "prompt", backend="jax")


import pytest
