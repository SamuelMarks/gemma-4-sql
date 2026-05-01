import pytest

from gemma_4_sql.sdk.chat import chat_turn


def test_chat_turn_routing():
    for backend in ["jax", "keras", "maxtext", "pytorch"]:
        res = chat_turn("foo", [], "prompt", backend=backend)
        assert res["backend"] == backend
        assert res["model"] == "foo"
        assert len(res["history"]) == 2


def test_chat_turn_routing_error():
    with pytest.raises(ValueError):
        chat_turn("foo", [], "prompt", backend="unknown")
