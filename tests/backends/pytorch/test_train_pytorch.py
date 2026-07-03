"""Tests for PyTorch training pipeline."""

import pytest

import gemma_4_sql.backends.pytorch.train as tr
from gemma_4_sql.backends.pytorch.train import train_model


class MockTensor:
    """Initialize class MockTensor."""

    def __init__(self, shape: object, _dtype: object = None, device: object = None) -> None:
        """Initialize function __init__.

        Args:
        ----
        shape: Description of shape.
        dtype: Description of dtype.
        device: Description of device.

        """
        self.shape = shape
        self.device = device

    def to(self, _device: object) -> object:
        """Initialize function to.

        Args:
        ----
        device: Description of device.

        """
        return self

    def view(self, *_args: object) -> object:
        """Initialize function view.

        Args:
        ----
        args: Description of args.

        """
        return self

    def size(self, *_args: object) -> object:
        """Initialize function size.

        Args:
        ----
        args: Description of args.

        """
        return 1

    def backward(self) -> object:
        """Initialize function backward."""

    def item(self) -> object:
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

            def zero_grad(self) -> object:
                """Initialize function zero_grad."""

            def step(self) -> object:
                """Initialize function step."""

        return MockOptimizer()


class MockGemma4ForCausalLM:
    """Initialize class MockGemma4ForCausalLM."""

    @classmethod
    def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
        """Initialize function from_pretrained.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """

        class MockModel:
            """Initialize class MockModel."""

            def __call__(self, *_args: object, **_kwargs: object) -> object:
                """Initialize function __call__.

                Args:
                ----
                args: Description of args.
                kwargs: Description of kwargs.

                """
                return MockTensor((1,))

            def to(self, _device: object) -> object:
                """Initialize function to.

                Args:
                ----
                device: Description of device.

                """
                return self

            def train(self) -> object:
                """Initialize function train."""

            def parameters(self) -> object:
                """Initialize function parameters."""
                return []

        return MockModel()


@pytest.fixture
def _mock_torch_env(monkeypatch: object) -> object:
    """Initialize function mock_torch_env.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(tr, "torch", MockTorch())
    monkeypatch.setattr(tr, "nn", MockNN())
    monkeypatch.setattr(tr, "optim", MockOptim())
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", MockGemma4ForCausalLM)

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Initialize function mock_build_dataloader.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        return {"loader": [{"inputs": MockTensor((1,)), "targets": MockTensor((1,))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)


@pytest.mark.usefixtures("_mock_torch_env")
def test_train_model_pytorch_real() -> object:
    """Initialize function test_train_model_pytorch_real.

    Args:
    ----
    mock_torch_env: Description of mock_torch_env.

    """
    res = train_model("sft", "mod", "dat", 2, 0.1)
    if not res["backend"] == "pytorch":
        raise AssertionError


def test_train_model_pytorch_missing() -> object:
    """Initialize function test_train_model_pytorch_missing."""
    orig_torch = tr.torch
    tr.torch = None
    res = train_model("sft", "mod", "dat", 2, 0.1)
    if not res["status"] == "mocked_missing_torch":
        raise AssertionError
    tr.torch = orig_torch


@pytest.mark.usefixtures("_mock_torch_env")
def test_train_model_pytorch_error(monkeypatch: object) -> object:
    """Initialize function test_train_model_pytorch_error.

    Args:
    ----
    mock_torch_env: Description of mock_torch_env.
    monkeypatch: Description of monkeypatch.

    """

    def mock_raise_error(*_args: object, **_kwargs: object) -> object:
        """Initialize function Exception.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(tr, "build_dataloader", Exception)
    train_model("sft", "mod", "dat", 2, 0.1)


@pytest.mark.usefixtures("_mock_torch_env")
def test_train_model_pytorch_no_loader_fallback(monkeypatch: object) -> object:
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

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    train_model("sft", "mod", "dat", 2, 0.1)


class MockModelObj:
    """Provide class docstring."""

    def to(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return self

    def parameters(self) -> object:
        """Execute function."""
        return [1]

    def train(self) -> None:
        """Execute function."""

    def __call__(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return type("Out", (), {"view": lambda _self, *_a: _self, "size": lambda _self, *_a: 1, "logits": type("L", (), {"view": lambda _self, *_a: _self, "size": lambda _self, *_a: 1})()})()

    def view(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return self

    def size(self, *_args: object, **_kwargs: object) -> int:
        """Execute function."""
        return 1


class MockModel:
    """Provide class docstring."""

    @classmethod
    def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return MockModelObj()


def test_train_model_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    m_train = __import__("gemma_4_sql.backends.pytorch.train")
    monkeypatch.setattr(m_train, "torch", MockTorch)
    monkeypatch.setattr(m_train, "optim", MockOptim)
    monkeypatch.setattr(m_train, "nn", MockNN)
    monkeypatch.setattr(m_train, "Gemma4ForCausalLM", MockModel)
    monkeypatch.setattr(m_train, "build_dataloader", lambda *_args, **_kwargs: {})
    res = m_train.train_model("sft", "m", "ds", 1, 0.1)
    if res["status"] != "completed":
        raise AssertionError
    monkeypatch.setattr(m_train, "build_dataloader", lambda *_args, **_kwargs: {"loader": [{"inputs": MockModelObj(), "targets": MockModelObj()}]})
    res = m_train.train_model("sft", "m", "ds", 1, 0.1)
    if res["status"] != "completed":
        raise AssertionError
