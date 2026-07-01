"""Module docstring."""

import pytest

from gemma_4_sql.backends.keras import chat


def test_chat_turn_keras(monkeypatch: pytest.MonkeyPatch) -> object:  # type: ignore[return]
    monkeypatch.setattr(chat, "tf", object())

    def mock_generate(*args: object, **kwargs: object) -> dict:
        return {"sql": "SELECT 1"}

    monkeypatch.setattr(chat, "generate_sql", mock_generate)

    res = chat.chat_turn("foo", [{"role": "user", "content": "hi"}], "how are you?")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["model"] == "foo":
        raise AssertionError
    if not res["status"] == "success_keras_chat":
        raise AssertionError
    if not res["response"] == "SELECT 1":
        raise AssertionError


def test_chat_turn_keras_missing(monkeypatch: pytest.MonkeyPatch) -> object:  # type: ignore[return]
    monkeypatch.setattr(chat, "tf", None)
    res = chat.chat_turn("foo", [], "prompt")
    if not res["status"] == "mocked_missing_keras":
        raise AssertionError


def test_chat_turn_keras_error(monkeypatch: pytest.MonkeyPatch) -> object:  # type: ignore[return]
    monkeypatch.setattr(chat, "tf", object())

    def raise_err(*args: object, **kwargs: object) -> dict:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(chat, "generate_sql", raise_err)

    res = chat.chat_turn("foo", [{"role": "user", "content": "hi"}], "how are you?")
    if "failed" not in str(res["status"]):
        raise AssertionError
