"""Tests for Keras inference logic."""

from __future__ import annotations

import builtins

import pytest

import gemma_4_sql.backends.keras.inference as inf
from gemma_4_sql.exceptions import DependencyMissingError


class MockTf:
    """Provide class docstring."""


def test_generate_sql_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.


    Raises:
        AssertionError: Description.

        TypeError: Description.

    """
    monkeypatch.setattr(inf, "tf", MockTf())
    monkeypatch.setattr(inf, "keras", object())
    res = inf.generate_sql("mock-model", "test prompt", beam_width=2, max_length=3, test_mode=True)
    if not res["status"] == "success":
        raise AssertionError
    if not res["backend"] == "keras":
        raise AssertionError
    if not isinstance(res["sql"], str):
        raise TypeError


def test_generate_sql_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(inf, "keras", None)
    with pytest.raises(DependencyMissingError):
        inf.generate_sql("mock-model", "test prompt")


def test_generate_sql_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(inf, "tf", MockTf())
    monkeypatch.setattr(inf, "keras", object())
    orig_import = __import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        if name == "keras_nlp.models":

            class MockGemma:
                """Provide class docstring."""

                @staticmethod
                def from_preset(*_args: object, **_kwargs: object) -> object:
                    """Execute function.

                    Returns:
                        object: Description of return.

                    """

                    class Model:
                        """Provide class docstring."""

                        def generate(self, *_args: object, **_kwargs: object) -> object:
                            """Execute function.

                            Raises:
                                RuntimeError: Description.

                            """
                            msg = "err"
                            raise RuntimeError(msg)

                    return Model()

            class MockModule:
                """Provide class docstring."""

                GemmaCausalLM = MockGemma

            return MockModule()
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)
    res = inf.generate_sql("mock-model", "test prompt", test_mode=False)
    if "failed" not in str(res["status"]):
        raise AssertionError


def test_inference_keras_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    mdl = __import__("gemma_4_sql.backends.keras.inference", fromlist=[""])
    monkeypatch.setitem(sys.modules, "keras", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    importlib.reload(mdl)


class MockKerasModel:
    """Provide class docstring."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Execute function."""

    def generate(self, prompt: object, max_length: object = None) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return prompt + "SELECT 1"

    @classmethod
    def from_preset(cls, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return cls()


def test_inference_keras_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    sys = __import__("sys", fromlist=[""])
    mdl = __import__("gemma_4_sql.backends.keras.inference", fromlist=[""])
    monkeypatch.setattr(mdl, "keras", type("MockKeras", (), {}))
    monkeypatch.setattr(mdl, "tf", type("MockTf", (), {}))
    monkeypatch.setitem(sys.modules, "keras_nlp", type("MockKerasNLP", (), {}))
    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("MockModels", (), {"GemmaCausalLM": MockKerasModel}))

    orig_import = builtins.__import__

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        if name == "keras_nlp.models" and fromlist and "GemmaCausalLM" in fromlist:
            return sys.modules["keras_nlp.models"]
        return orig_import(name, _globals, _locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)
    res = mdl.generate_sql("model", "prompt")
    if res["sql"] != "SELECT 1":
        raise AssertionError
