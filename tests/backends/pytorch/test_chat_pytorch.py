"""Tests for PyTorch chat logic."""

from __future__ import annotations

import typing

import gemma_4_sql.backends.pytorch.chat as pt_chat
from gemma_4_sql.backends.pytorch.chat import chat_turn

if typing.TYPE_CHECKING:
    import pytest


class MockTorch:
    pass


class MockAutoTokenizer:
    pass


def test_chat_turn_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt_chat, "torch", MockTorch())
    monkeypatch.setattr(pt_chat, "AutoTokenizer", MockAutoTokenizer())

    res = chat_turn("foo", [{"role": "user", "content": "hi"}], "how are you?", test_mode=True)
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["model"] == "foo":
        raise AssertionError
    if not res["status"] == "success_pytorch_chat":
        raise AssertionError


def test_chat_turn_pytorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt_chat, "torch", None)
    res = chat_turn("foo", [], "prompt")
    if not res["status"] == "mocked_missing_pytorch":
        raise AssertionError


def test_chat_turn_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt_chat, "torch", MockTorch())
    monkeypatch.setattr(pt_chat, "AutoTokenizer", MockAutoTokenizer())

    def raise_err(*args: object, **kwargs: object) -> object:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(pt_chat, "generate_sql", raise_err)

    res = chat_turn("foo", [], "prompt", test_mode=False)
    if "failed" not in str(res["status"]):
        raise AssertionError
