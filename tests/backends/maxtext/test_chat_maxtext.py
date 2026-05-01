import sys
from unittest import mock

import gemma_4_sql.backends.maxtext.chat as chat


def test_chat_turn_maxtext():
    res = chat.chat_turn("foo", [{"role": "user", "content": "hi"}], "how are you?")
    assert res["backend"] == "maxtext"
    assert res["model"] == "foo"

def test_chat_turn_maxtext_missing():
    with mock.patch.dict(sys.modules, {"maxtext.models.gemma4": None, "maxtext.models": None, "maxtext": None}):
        import importlib
        importlib.reload(chat)
        res = chat.chat_turn("foo", [], "prompt")
        assert res["status"] == "mocked_missing_maxtext"
