"""Module docstring."""

import sys
from unittest import mock

from gemma_4_sql.backends.maxtext import chat


def test_chat_turn_maxtext() -> object:  # type: ignore[return]
    """Initialize function test_chat_turn_maxtext."""
    res = chat.chat_turn("foo", [{"role": "user", "content": "hi"}], "how are you?")
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["model"] == "foo":
        raise AssertionError


def test_chat_turn_maxtext_missing() -> object:  # type: ignore[return]
    """Initialize function test_chat_turn_maxtext_missing."""
    with mock.patch.dict(sys.modules, {"maxtext.models.gemma4": None, "maxtext.models": None, "maxtext": None}):
        importlib = __import__("importlib")
        importlib.reload(chat)
        res = chat.chat_turn("foo", [], "prompt")
        if not res["status"] == "mocked_missing_maxtext":
            raise AssertionError
