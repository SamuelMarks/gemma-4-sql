import sys
from unittest import mock

import gemma_4_sql.backends.pytorch.chat as chat


def test_chat_turn_pytorch():
    res = chat.chat_turn("foo", [{"role": "user", "content": "hi"}], "how are you?")
    assert res["backend"] == "pytorch"
    assert res["model"] == "foo"

def test_chat_turn_pytorch_missing():
    with mock.patch.dict(sys.modules, {"torch": None}):
        import importlib
        importlib.reload(chat)
        res = chat.chat_turn("foo", [], "prompt")
        assert res["status"] == "mocked_missing_pytorch"
