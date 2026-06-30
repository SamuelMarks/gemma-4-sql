"""Module docstring."""

import sys
from unittest import mock

import pytest

from gemma_4_sql.backends.maxtext import chat


def test_chat_turn_maxtext(monkeypatch: pytest.MonkeyPatch) -> object:  # type: ignore[return]
    """Initialize function test_chat_turn_maxtext."""
    monkeypatch.setattr(chat, "gemma4", object())

    def mock_generate(*args: object, **kwargs: object) -> dict:
        return {"sql": "SELECT 1"}

    monkeypatch.setattr(chat, "generate_sql", mock_generate)

    res = chat.chat_turn("foo", [{"role": "user", "content": "hi"}], "how are you?")
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["model"] == "foo":
        raise AssertionError
    if not res["status"] == "success_maxtext_chat":
        raise AssertionError
    if not res["response"] == "SELECT 1":
        raise AssertionError


def test_chat_turn_maxtext_missing() -> object:  # type: ignore[return]
    """Initialize function test_chat_turn_maxtext_missing."""
    with mock.patch.dict(sys.modules, {"maxtext.models.gemma4": None, "maxtext.models": None, "maxtext": None}):
        importlib = __import__("importlib")
        importlib.reload(chat)
        res = chat.chat_turn("foo", [], "prompt")
        if not res["status"] == "mocked_missing_maxtext":
            raise AssertionError


def test_chat_turn_maxtext_error(monkeypatch: pytest.MonkeyPatch) -> object:  # type: ignore[return]
    """Test error."""
    monkeypatch.setattr(chat, "gemma4", object())

    def mock_generate(*args: object, **kwargs: object) -> dict:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(chat, "generate_sql", mock_generate)

    res = chat.chat_turn("foo", [{"role": "user", "content": "hi"}], "how are you?")
    if "failed" not in str(res["status"]):
        raise AssertionError
