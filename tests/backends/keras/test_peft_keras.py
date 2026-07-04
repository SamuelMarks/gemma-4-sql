"""Tests for Keras PEFT."""

from __future__ import annotations

import pytest

import gemma_4_sql.backends.keras.peft as pt


class MockLayers:
    """Provide class docstring."""

    @staticmethod
    def mock_embedding(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return lambda _x: "x"

    Embedding = mock_embedding

    @staticmethod
    def mock_dense(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return lambda _x: "x"

    Dense = mock_dense


class MockModel:
    """Provide class docstring."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Execute function."""
        self.backbone = self

    def enable_lora(self, rank: int | None = None) -> None:
        """Execute function."""
        self.lora_enabled = True


class MockKeras:
    """Provide class docstring."""

    @staticmethod
    def mock_input(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "inputs"

    Input = mock_input
    Model = MockModel
    layers = MockLayers()


def test_apply_lora_keras_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt, "keras", None)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "mocked_missing_keras":
        raise AssertionError
    if not res["backend"] == "keras":
        raise AssertionError


def test_apply_lora_keras_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt, "keras", MockKeras())
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "completed":
        raise AssertionError


def test_apply_lora_keras_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt, "keras", MockKeras())

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockKeras, "Input", raise_err)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if "failed" not in str(res["status"]):
        raise AssertionError


def test_peft_keras_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    mdl = __import__("gemma_4_sql.backends.keras.peft", fromlist=[""])
    monkeypatch.setitem(sys.modules, "keras", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    importlib.reload(mdl)


class MockKerasModel:
    """Provide class docstring."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Execute function."""

    @classmethod
    def from_preset(cls, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return cls()


def test_apply_lora_keras_real_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    sys = __import__("sys", fromlist=[""])
    mdl = __import__("gemma_4_sql.backends.keras.peft", fromlist=[""])
    monkeypatch.setattr(mdl, "keras", type("MockKeras", (), {}))
    monkeypatch.setitem(sys.modules, "keras_nlp", type("MockKerasNLP", (), {}))
    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("MockModels", (), {"GemmaCausalLM": MockKerasModel}))

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        if name == "keras_nlp.models" and "GemmaCausalLM" in fromlist:
            return sys.modules["keras_nlp.models"]
        builtins = __import__("builtins", fromlist=[""])
        return builtins.__import__(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)
    mdl.apply_lora("model", ["q_proj"], 8, 16, 0.05)


def test_apply_lora_keras_mock_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    mdl = __import__("gemma_4_sql.backends.keras.peft", fromlist=[""])
    monkeypatch.setattr(mdl, "keras", MockKeras())
    builtins = __import__("builtins", fromlist=[""])
    orig_import = builtins.__import__

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Execute function.

        Returns:
            object: Description of return.


        Raises:
            ImportError: Description.

        """
        if name == "keras_nlp.models":
            msg = "mock"
            raise ImportError(msg)
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)
    mdl.apply_lora("model", ["q_proj"], 8, 16, 0.05)
