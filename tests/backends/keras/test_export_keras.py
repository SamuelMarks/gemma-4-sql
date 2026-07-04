# Copyright 2024
"""Provide module docstring."""

import contextlib
from typing import NoReturn as Never

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
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(kexp, "keras", MockKeras())
    res = kexp.export_model("model", "out")
    if res["backend"] != "keras":
        raise AssertionError


def test_export_keras_error(monkeypatch: object) -> None:
    """Execute function."""
    monkeypatch.setattr(kexp, "keras", MockKeras())

    def raise_err(*_args: object, **_kwargs: object) -> Never:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockKeras.Model, "save", raise_err)
    with contextlib.suppress(ValueError):
        kexp.export_model("model", "out")


def test_export_keras_missing(monkeypatch: object) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(kexp, "keras", None)
    res = kexp.export_model("model", "out")
    if res["status"] != "mock_exported":
        raise AssertionError


def test_export_keras_imports_fail(monkeypatch: object) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "keras", None)
    importlib.reload(kexp)
    monkeypatch.undo()
    importlib.reload(kexp)
