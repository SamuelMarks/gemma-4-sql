"""Module docstring."""

import pytest

import gemma_4_sql.sdk.chat as sdk_chat


def test_chat_turn_routing(monkeypatch: pytest.MonkeyPatch) -> object:
    """Initialize function test_chat_turn_routing."""
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, _globals=None, _locals=None, fromlist=(), level=0):
        if name == "gemma_4_sql.sdk.registry" and "get_backend" in fromlist:

            class MockBackend:
                def __init__(self, backend_name):
                    self.backend_name = backend_name

                def chat_turn(self, model_name, history, new_prompt, **kwargs):
                    return {"backend": self.backend_name, "model": model_name, "history": [1, 2]}

            return type("M", (), {"get_backend": MockBackend})
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)

    for backend in ["jax", "keras", "maxtext", "pytorch"]:
        res = sdk_chat.chat_turn("foo", [], "prompt", backend=backend)
        if not res["backend"] == backend:
            raise AssertionError
        if not res["model"] == "foo":
            raise AssertionError
        if not len(res["history"]) == 2:
            raise AssertionError


def test_chat_turn_routing_error(monkeypatch: pytest.MonkeyPatch) -> object:
    """Initialize function test_chat_turn_routing_error."""
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, _globals=None, _locals=None, fromlist=(), level=0):
        if name == "gemma_4_sql.sdk.registry" and "get_backend" in fromlist:

            def raise_err(b):
                msg = "err"
                raise ValueError(msg)

            return type("M", (), {"get_backend": raise_err})
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)

    with pytest.raises(ValueError, match=r".*"):
        sdk_chat.chat_turn("foo", [], "prompt", backend="unknown")
