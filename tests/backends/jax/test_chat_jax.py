"""Module docstring."""

import sys
from unittest import mock

from gemma_4_sql.backends.jax import chat


def test_chat_turn_jax() -> object:  # type: ignore[return]
    """Initialize function test_chat_turn_jax."""
    res = chat.chat_turn("foo", [{"role": "user", "content": "hi"}], "how are you?")
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["model"] == "foo":
        raise AssertionError
    if not len(res["history"]) == int("3"):  # type: ignore[arg-type]
        raise AssertionError
    if not res["history"][-1]["role"] == "assistant":  # type: ignore[index]
        raise AssertionError
    if "how are you?" not in res["response"]:  # type: ignore[operator]
        raise AssertionError


def test_chat_turn_jax_missing() -> object:  # type: ignore[return]
    """Initialize function test_chat_turn_jax_missing."""
    with mock.patch.dict(sys.modules, {"jax": None}):
        importlib = __import__("importlib")
        importlib.reload(chat)
        res = chat.chat_turn("foo", [], "prompt")
        if not res["status"] == "mocked_missing_jax":
            raise AssertionError
