"""Provide module docstring."""

import pytest

import gemma_4_sql.backends.keras.export as kexp


class MockKeras:
    """Provide class docstring."""

    class Model:
        """Provide class docstring."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Execute function."""

        def save(self, path: object) -> None:
            """Execute function."""

    def mock_input(self, *_args: object, **_kwargs: object) -> str:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "input"

    Input = mock_input

    class MockLayers:
        """Provide class docstring."""

        def mock_embedding(*_args: object, **_kwargs: object) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return lambda _x: "x"

        Embedding = mock_embedding

        def mock_dense(*_args: object, **_kwargs: object) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return lambda _x: "x"

        Dense = mock_dense

    layers = MockLayers


def test_export_keras_real(monkeypatch: object) -> None:
    """Execute function."""
    monkeypatch.setattr(kexp, "keras", MockKeras())

    builtins = __import__("builtins", fromlist=[""])
    orig_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Docstring."""
        if name == "keras_nlp.models":

            class MockGemma:
                """Docstring."""

                @staticmethod
                def from_preset(*args, **kwargs):
                    """Docstring."""
                    return MockKeras.Model()

            return type("M", (), {"GemmaCausalLM": MockGemma})()
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)

    res = kexp.export_model("model", "out")
    if res["backend"] != "keras":
        raise AssertionError


def test_export_keras_error(monkeypatch: object) -> None:
    """Execute function."""
    monkeypatch.setattr(kexp, "keras", MockKeras())

    with pytest.raises(ValueError, match="Failed to load model model"):
        kexp.export_model("model", "out")


def test_export_keras_missing(monkeypatch: object) -> None:
    """Execute function."""
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(kexp, "keras", None)
    with pytest.raises(DependencyMissingError):
        kexp.export_model("model", "out")


def test_export_keras_imports_fail(monkeypatch: object) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "keras", None)
    importlib.reload(kexp)
    monkeypatch.undo()
    importlib.reload(kexp)
