"""Module docstring."""

import sys
from unittest import mock

from gemma_4_sql.backends.keras import chat


def test_chat_turn_keras() -> object:  # type: ignore[return]
    """Initialize function test_chat_turn_keras."""
    res = chat.chat_turn("foo", [{"role": "user", "content": "hi"}], "how are you?")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["model"] == "foo":
        raise AssertionError
    if not len(res["history"]) == int("3"):  # type: ignore[arg-type]
        raise AssertionError


def test_chat_turn_keras_missing() -> object:  # type: ignore[return]
    """Initialize function test_chat_turn_keras_missing."""
    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        importlib = __import__("importlib")
        importlib.reload(chat)
        res = chat.chat_turn("foo", [], "prompt")
        if not res["status"] == "mocked_missing_keras":
            raise AssertionError
