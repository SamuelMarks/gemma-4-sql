"""Tests for Keras training pipeline."""

import typing

import pytest

import gemma_4_sql.backends.keras.train as tr
from gemma_4_sql.backends.keras.train import train_model


class MockTfTensor:
    """Initialize class MockTfTensor."""

    def __init__(self: typing.Any, shape: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        shape: Description of shape.

        """
        self.shape = shape


class MockTf:
    """Initialize class MockTf."""

    int32 = 1

    @staticmethod
    def zeros(shape: object, _dtype: object = None) -> object:
        """Initialize function zeros.

        Args:
        ----
        shape: Description of shape.
        dtype: Description of dtype.

        """
        return MockTfTensor(shape)


class MockHistory:
    """Initialize class MockHistory."""

    def __init__(self: typing.Any) -> None:
        """Initialize function __init__."""
        self.history = {"loss": [0.5, 0.1]}


class MockModel:
    """Initialize class MockModel."""

    def compile(self: typing.Any, *args: object, **kwargs: object) -> object:
        """Initialize function compile.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """

    def fit(self: typing.Any, *_args: object, **_kwargs: object) -> object:
        """Initialize function fit.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        return MockHistory()


class MockLayers:
    """Initialize class MockLayers."""

    @staticmethod
    def embedding(*_args: object, **_kwargs: object) -> object:
        """Initialize function embedding.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        return lambda x: x

    @staticmethod
    def dense(*_args: object, **_kwargs: object) -> object:
        """Initialize function dense.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        return lambda x: x


class MockOptimizers:
    """Initialize class MockOptimizers."""

    @staticmethod
    def adamw(*_args: object, **_kwargs: object) -> object:
        """Initialize function adamw.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        return "adamw"


class MockLosses:
    """Initialize class MockLosses."""

    @staticmethod
    def sparsecategoricalcrossentropy(*_args: object, **_kwargs: object) -> object:
        """Initialize function sparsecategoricalcrossentropy.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        return "loss"


class MockKeras:
    """Initialize class MockKeras."""

    Input = staticmethod(lambda *_args, **_kwargs: "input")
    Model = staticmethod(lambda *_args, **_kwargs: MockModel())
    layers = MockLayers()
    optimizers = MockOptimizers()
    losses = MockLosses()


@pytest.fixture
def _mock_keras_env(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function mock_keras_env.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(tr, "keras", MockKeras())  # type: ignore[attr-defined]
    monkeypatch.setattr(tr, "tf", MockTf())  # type: ignore[attr-defined]

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Initialize function mock_build_dataloader.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        return {"loader": [{"inputs": MockTfTensor((1,)), "targets": MockTfTensor((1,))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)  # type: ignore[attr-defined]


@pytest.mark.usefixtures("_mock_keras_env")
def test_train_model_keras_real() -> object:  # type: ignore[return]
    """Initialize function test_train_model_keras_real.

    Args:
    ----
    mock_keras_env: Description of mock_keras_env.

    """
    res = train_model("sft", "mod", "dat", 2, 0.1)
    if not res["backend"] == "keras":
        raise AssertionError


def test_train_model_keras_missing() -> object:  # type: ignore[return]
    """Initialize function test_train_model_keras_missing."""
    orig_keras = tr.keras  # type: ignore[attr-defined]
    orig_tf = tr.tf  # type: ignore[attr-defined]
    tr.keras = None  # type: ignore[attr-defined]
    tr.tf = None  # type: ignore[attr-defined]
    res = train_model("sft", "mod", "dat", 2, 0.1)
    if not res["status"] == "mocked_missing_keras":
        raise AssertionError
    model = tr.KerasSQLModel()
    if model(None) is not None:
        raise AssertionError
    tr.keras = orig_keras  # type: ignore[attr-defined]
    tr.tf = orig_tf  # type: ignore[attr-defined]


@pytest.mark.usefixtures("_mock_keras_env")
def test_train_model_keras_error(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function test_train_model_keras_error.

    Args:
    ----
    mock_keras_env: Description of mock_keras_env.
    monkeypatch: Description of monkeypatch.

    """

    def raise_error(*_args: object, **_kwargs: object) -> object:
        """Initialize function raise_error.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(tr, "build_dataloader", raise_error)  # type: ignore[attr-defined]
    train_model("sft", "mod", "dat", 2, 0.1)


@pytest.mark.usefixtures("_mock_keras_env")
def test_train_model_keras_no_loader_fallback(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function test_train_model_keras_no_loader_fallback.

    Args:
    ----
    mock_keras_env: Description of mock_keras_env.
    monkeypatch: Description of monkeypatch.

    """

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Initialize function mock_build_dataloader.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        return {"loader": None}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)  # type: ignore[attr-defined]
    train_model("sft", "mod", "dat", 2, 0.1)
