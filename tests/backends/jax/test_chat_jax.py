"""Module docstring."""

import sys
from unittest import mock

import pytest

from gemma_4_sql.backends.jax import chat


def test_chat_turn_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function test_chat_turn_jax."""
    monkeypatch.setattr(chat, "jax", object())

    def mock_generate_sql(model_name: str, prompt: str, **kwargs: object) -> dict:
        return {"sql": "SELECT * FROM generated_chat"}

    monkeypatch.setattr(chat, "generate_sql", mock_generate_sql)

    res = chat.chat_turn("foo", [{"role": "user", "content": "hi"}], "how are you?")
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["model"] == "foo":
        raise AssertionError
    if not len(res["history"]) == 3:
        raise AssertionError
    if not res["response"] == "SELECT * FROM generated_chat":
        raise AssertionError


def test_chat_turn_jax_missing() -> None:
    """Initialize function test_chat_turn_jax_missing."""
    with mock.patch.dict(sys.modules, {"jax": None}):
        importlib = __import__("importlib")
        importlib.reload(chat)
        res = chat.chat_turn("foo", [], "prompt")
        if not res["status"] == "mocked_missing_jax":
            raise AssertionError
        if not res["response"] == "SELECT * FROM fallback_chat":
            raise AssertionError
