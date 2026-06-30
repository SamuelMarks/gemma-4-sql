"""Tests for PyTorch training pipeline."""

import typing

import gemma_4_sql.backends.pytorch.train as tr
import pytest
from gemma_4_sql.backends.pytorch.train import train_model


class MockTensor:
    """Initialize class MockTensor."""

    def __init__(self: typing.Any, shape: object, _dtype: object = None, device: object = None) -> None:
        """Initialize function __init__.

        Args:
        ----
        shape: Description of shape.
        dtype: Description of dtype.
        device: Description of device.

        """
        self.shape = shape
        self.device = device

    def to(self: typing.Any, _device: object) -> object:
        """Initialize function to.

        Args:
        ----
        device: Description of device.

        """
        return self

    def view(self: typing.Any, *_args: object) -> object:
        """Initialize function view.

        Args:
        ----
        args: Description of args.

        """
        return self

    def size(self: typing.Any, *_args: object) -> object:
        """Initialize function size.

        Args:
        ----
        args: Description of args.

        """
        return 1

    def backward(self: typing.Any) -> object:
        """Initialize function backward."""

    def item(self: typing.Any) -> object:
        """Initialize function item."""
        return 0.1


class MockCuda:
    """Initialize class MockCuda."""

    @staticmethod
    def is_available() -> object:
        """Initialize function is_available."""
        return False


class MockTorch:
    """Initialize class MockTorch."""

    Tensor = MockTensor
    long = 1
    cuda = MockCuda()

    @staticmethod
    def zeros(shape: object, dtype: object = None, device: object = None) -> object:
        """Initialize function zeros.

        Args:
        ----
        shape: Description of shape.
        dtype: Description of dtype.
        device: Description of device.

        """
        return MockTensor(shape, dtype, device)

    @staticmethod
    def device(name: object) -> object:
        """Initialize function device.

        Args:
        ----
        name: Description of name.

        """
        return name


class MockNN:
    """Initialize class MockNN."""

    @staticmethod
    def crossentropyloss() -> object:
        """Initialize function crossentropyloss."""

        def loss_fn(*_args: object, **_kwargs: object) -> object:
            """Initialize function loss_fn.

            Args:
            ----
            args: Description of args.
            kwargs: Description of kwargs.

            """
            return MockTensor((1,))

        return loss_fn


class MockOptim:
    """Initialize class MockOptim."""

    @staticmethod
    def adamw(*_args: object, **_kwargs: object) -> object:
        """Initialize function adamw.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """

        class MockOptimizer:
            """Initialize class MockOptimizer."""

            def zero_grad(self: typing.Any) -> object:
                """Initialize function zero_grad."""

            def step(self: typing.Any) -> object:
                """Initialize function step."""

        return MockOptimizer()


class MockGemma4ForCausalLM:
    """Initialize class MockGemma4ForCausalLM."""

    @classmethod
    def from_pretrained(cls: typing.Any, *_args: object, **_kwargs: object) -> object:
        """Initialize function from_pretrained.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """

        class MockModel:
            """Initialize class MockModel."""

            def __call__(self: typing.Any, *_args: object, **_kwargs: object) -> object:
                """Initialize function __call__.

                Args:
                ----
                args: Description of args.
                kwargs: Description of kwargs.

                """
                return MockTensor((1,))

            def to(self: typing.Any, _device: object) -> object:
                """Initialize function to.

                Args:
                ----
                device: Description of device.

                """
                return self

            def train(self: typing.Any) -> object:
                """Initialize function train."""

            def parameters(self: typing.Any) -> object:
                """Initialize function parameters."""
                return []

        return MockModel()


@pytest.fixture()
def _mock_torch_env(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function mock_torch_env.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(tr, "torch", MockTorch())  # type: ignore[attr-defined]
    monkeypatch.setattr(tr, "nn", MockNN())  # type: ignore[attr-defined]
    monkeypatch.setattr(tr, "optim", MockOptim())  # type: ignore[attr-defined]
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", MockGemma4ForCausalLM)  # type: ignore[attr-defined]

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Initialize function mock_build_dataloader.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        return {"loader": [{"inputs": MockTensor((1,)), "targets": MockTensor((1,))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)  # type: ignore[attr-defined]


@pytest.mark.usefixtures("_mock_torch_env")
def test_train_model_pytorch_real() -> object:  # type: ignore[return]
    """Initialize function test_train_model_pytorch_real.

    Args:
    ----
    mock_torch_env: Description of mock_torch_env.

    """
    res = train_model("sft", "mod", "dat", 2, 0.1)
    if not res["backend"] == "pytorch":
        raise AssertionError


def test_train_model_pytorch_missing() -> object:  # type: ignore[return]
    """Initialize function test_train_model_pytorch_missing."""
    orig_torch = tr.torch  # type: ignore[attr-defined]
    tr.torch = None  # type: ignore[attr-defined]
    res = train_model("sft", "mod", "dat", 2, 0.1)
    if not res["status"] == "mocked_missing_torch":
        raise AssertionError
    tr.torch = orig_torch  # type: ignore[attr-defined]


@pytest.mark.usefixtures("_mock_torch_env")
def test_train_model_pytorch_error(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function test_train_model_pytorch_error.

    Args:
    ----
    mock_torch_env: Description of mock_torch_env.
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


@pytest.mark.usefixtures("_mock_torch_env")
def test_train_model_pytorch_no_loader_fallback(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function test_train_model_pytorch_no_loader_fallback.

    Args:
    ----
    mock_torch_env: Description of mock_torch_env.
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
