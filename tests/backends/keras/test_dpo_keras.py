# Copyright 2024
"""Tests for Keras DPO logic."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import gemma_4_sql.backends.keras.dpo as kr_dpo
from gemma_4_sql.backends.keras.dpo import run_dpo
from gemma_4_sql.type_hints import DPOConfig

if TYPE_CHECKING:
    import pytest


class MockTensor:
    """Provide class docstring."""

    def __sub__(self: object, other: object) -> MockTensor:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def __mul__(self: object, other: object) -> MockTensor:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def __rmul__(self: object, other: object) -> MockTensor:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def __neg__(self: typing.Any) -> MockTensor:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def numpy(self: typing.Any) -> float:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return 0.42


class MockMath:
    """Provide class docstring."""

    def log_sigmoid(self: object, _x: object) -> MockTensor:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockTensor()


class MockGradientTape:
    """Provide class docstring."""

    def __enter__(self) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self

    def __exit__(self, *args: object, **kwargs: object) -> None:
        """Execute function."""

    def gradient(self, *_args: object, **_kwargs: object) -> list:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return ["grads"]


def mock_dpo_loss(*_args: object, **_kwargs: object) -> object:
    """Execute function.

    Returns:
        object: Description of return.

    """
    return (MockTensor(), MockTensor(), MockTensor())


def test_dpo_loss() -> None:
    """Execute function."""
    mdl = __import__("gemma_4_sql.backends.keras.dpo", fromlist=[""])
    mdl.tf = MockTf()
    mdl.dpo_loss(MockTensor(), MockTensor(), MockTensor(), MockTensor())


class MockTf:
    """Provide class docstring."""

    float32 = "float32"
    int32 = "int32"

    def __init__(self: typing.Any) -> None:
        """Execute function."""
        self.math = MockMath()

    def reduce_sum(self: object, *_args: object, **_kwargs: object) -> MockTensor:
        """Mock reduce_sum.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def reduce_mean(self: object, *_args: object, **_kwargs: object) -> MockTensor:
        """Mock reduce_mean.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def constant(self: object, *_args: object, **_kwargs: object) -> MockTensor:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def cast(self: object, *_args: object, **_kwargs: object) -> MockTensor:
        """Mock cast.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def zeros(self, *_args: object, **_kwargs: object) -> MockTensor:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def function(self, fn: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return fn

    def mock_gradienttape(self) -> MockGradientTape:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockGradientTape()

    GradientTape = mock_gradienttape


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

    layers = MockLayers

    class Model:
        """Provide class docstring."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """Execute function."""
            self.trainable_variables = ["vars"]

        def __call__(self, *_args: object, **_kwargs: object) -> MockTensor:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return MockTensor()

    class MockOptimizers:
        """Provide class docstring."""

        class AdamW:
            """Provide class docstring."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                """Execute function."""

            def apply_gradients(self, *args: object, **kwargs: object) -> None:
                """Execute function."""

    optimizers = MockOptimizers


def test_run_dpo_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras DPO when missing.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(kr_dpo, "tf", None)
    res = run_dpo(DPOConfig(model_name="model", dataset="data"))
    if not res["status"] == "mocked_missing_keras":
        raise AssertionError


def test_run_dpo_keras_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras DPO.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(kr_dpo, "tf", MockTf())
    monkeypatch.setattr(kr_dpo, "keras", MockKeras)

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> dict:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return {"loader": [{"chosen_inputs": MockTensor(), "chosen_labels": MockTensor(), "rejected_inputs": MockTensor(), "rejected_labels": MockTensor()}]}

    monkeypatch.setattr(kr_dpo, "build_dataloader", mock_build_dataloader)
    res = run_dpo(DPOConfig(model_name="model", dataset="data"))
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["status"] == "completed":
        raise AssertionError


def test_run_dpo_keras_no_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(kr_dpo, "tf", MockTf())
    monkeypatch.setattr(kr_dpo, "keras", MockKeras)

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> dict:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return {"loader": None}

    monkeypatch.setattr(kr_dpo, "build_dataloader", mock_build_dataloader)
    res = run_dpo(DPOConfig(model_name="model", dataset="data"))
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["status"] == "completed":
        raise AssertionError


def test_run_dpo_keras_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(kr_dpo, "tf", MockTf())
    monkeypatch.setattr(kr_dpo, "keras", MockKeras)

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> dict:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(kr_dpo, "build_dataloader", mock_build_dataloader)
    res = run_dpo(DPOConfig(model_name="model", dataset="data"))
    if "failed" not in str(res["status"]):
        raise AssertionError


def test_dpo_keras_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    mdl = __import__("gemma_4_sql.backends.keras.dpo", fromlist=[""])
    monkeypatch.setitem(sys.modules, "keras", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    importlib.reload(mdl)
