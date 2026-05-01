import sys
from unittest import mock

import gemma_4_sql.backends.keras.chat as chat


def test_chat_turn_keras():
    res = chat.chat_turn("foo", [{"role": "user", "content": "hi"}], "how are you?")
    assert res["backend"] == "keras"
    assert res["model"] == "foo"
    assert len(res["history"]) == 3


def test_chat_turn_keras_missing():
    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        import importlib

        importlib.reload(chat)
        res = chat.chat_turn("foo", [], "prompt")
        assert res["status"] == "mocked_missing_keras"
