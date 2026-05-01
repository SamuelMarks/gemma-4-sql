import sys
from unittest import mock

import gemma_4_sql.backends.jax.chat as chat


def test_chat_turn_jax():
    res = chat.chat_turn("foo", [{"role": "user", "content": "hi"}], "how are you?")
    assert res["backend"] == "jax"
    assert res["model"] == "foo"
    assert len(res["history"]) == 3
    assert res["history"][-1]["role"] == "assistant"
    assert "how are you?" in res["response"]


def test_chat_turn_jax_missing():
    with mock.patch.dict(sys.modules, {"jax": None}):
        import importlib

        importlib.reload(chat)
        res = chat.chat_turn("foo", [], "prompt")
        assert res["status"] == "mocked_missing_jax"
