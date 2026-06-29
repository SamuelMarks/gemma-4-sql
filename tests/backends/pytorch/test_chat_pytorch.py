"""Module docstring."""

import sys
from unittest import mock

from gemma_4_sql.backends.pytorch import chat


def test_chat_turn_pytorch() -> object:  # type: ignore[return]
    """Initialize function test_chat_turn_pytorch."""
    res = chat.chat_turn("foo", [{"role": "user", "content": "hi"}], "how are you?")
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["model"] == "foo":
        raise AssertionError


def test_chat_turn_pytorch_missing() -> object:  # type: ignore[return]
    """Initialize function test_chat_turn_pytorch_missing."""
    with mock.patch.dict(sys.modules, {"torch": None}):
        importlib = __import__("importlib")
        importlib.reload(chat)
        res = chat.chat_turn("foo", [], "prompt")
        if not res["status"] == "mocked_missing_pytorch":
            raise AssertionError
