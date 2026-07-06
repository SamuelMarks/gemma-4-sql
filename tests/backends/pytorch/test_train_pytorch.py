"""Tests for PyTorch training pipeline."""

import pytest

import gemma_4_sql.backends.pytorch.train as tr
from gemma_4_sql.backends.pytorch.train import train_model
from gemma_4_sql.type_hints import TrainingConfig


class MockTensor:
    """Initialize class MockTensor."""

    def __init__(self, shape: object, _dtype: object = None, device: object = None) -> None:
        """Initialize function __init__.

        Args:
        ----
        shape: Description of shape.
        device: Description of device.

        """
        self.shape = shape
        self.device = device

    def to(self, _device: object) -> object:
        """Initialize function to.

        Returns:
            object: Description of return.

        """
        return self

    def view(self, *_args: object) -> object:
        """Initialize function view.

        Args:
        ----
        args: Description of args.


        Returns:
            object: Description of return.

        """
        return self

    def size(self, *_args: object) -> object:
        """Initialize function size.

        Args:
        ----
        args: Description of args.


        Returns:
            object: Description of return.

        """
        return 1

    def backward(self) -> object:
        """Initialize function backward."""

    def item(self) -> object:
        """Initialize function item.

        Returns:
            object: Description of return.

        """
        return 0.1


class MockCuda:
    """Initialize class MockCuda."""

    @staticmethod
    def is_available() -> object:
        """Initialize function is_available.

        Returns:
            object: Description of return.

        """
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


        Returns:
            object: Description of return.

        """
        return MockTensor(shape, dtype, device)

    @staticmethod
    def device(name: object) -> object:
        """Initialize function device.

        Args:
        ----
        name: Description of name.


        Returns:
            object: Description of return.

        """
        return name


class MockNN:
    """Mock NN."""

    class CrossEntropyLoss:
        """Mock Loss."""

        def __init__(self, *args, **kwargs) -> None:
            """Init."""

        def __call__(self, *args, **kwargs):
            """Call.

            Returns:
                object: Description of return.

            """

            class T:
                """Docstring."""

                def backward(self) -> None:
                    """Docstring."""

                def item(self) -> float:
                    """Docstring."""
                    return 0.0

            return T()

    """Initialize class MockNN."""

    @staticmethod
    def crossentropyloss() -> object:
        """Initialize function crossentropyloss.

        Returns:
            object: Description of return.

        """

        def loss_fn(*_args: object, **_kwargs: object) -> object:
            """Initialize function loss_fn.

            Args:
            ----
            args: Description of args.
            kwargs: Description of kwargs.


            Returns:
                object: Description of return.

            """
            return MockTensor((1,))

        return loss_fn


class MockOptim:
    """Mock Optim."""

    class AdamW:
        """Mock Adam."""

        def __init__(self, *args, **kwargs) -> None:
            """Init."""

        def step(self) -> None:
            """Step."""

        def zero_grad(self) -> None:
            """Zero grad."""

    """Initialize class MockOptim."""

    @staticmethod
    def adamw(*_args: object, **_kwargs: object) -> object:
        """Initialize function adamw.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.


        Returns:
            object: Description of return.

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


        Returns:
            object: Description of return.

        """

        class MockModel:
            """Initialize class MockModel."""

            def __call__(self, *_args: object, **_kwargs: object) -> object:
                """Initialize function __call__.

                Args:
                ----
                args: Description of args.
                kwargs: Description of kwargs.


                Returns:
                    object: Description of return.

                """
                return MockTensor((1,))

            def to(self, _device: object) -> object:
                """Initialize function to.

                Returns:
                    object: Description of return.

                """
                return self

            def train(self) -> object:
                """Initialize function train."""

            def parameters(self) -> object:
                """Initialize function parameters.

                Returns:
                    object: Description of return.

                """
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


        Returns:
            object: Description of return.

        """
        return {"loader": [{"inputs": MockTensor((1,)), "targets": MockTensor((1,))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)


@pytest.mark.usefixtures("_mock_torch_env")
def test_train_model_pytorch_real() -> object:
    """Initialize function test_train_model_pytorch_real.

    Raises:
        AssertionError: Description.

    """
    res = train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1))
    if not res["backend"] == "pytorch":
        raise AssertionError


def test_train_model_pytorch_missing() -> object:
    """Initialize function test_train_model_pytorch_missing.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    orig_torch = tr.torch
    tr.torch = None
    with pytest.raises(DependencyMissingError, match="PyTorch dependencies are missing."):
        train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1))
    tr.torch = orig_torch


@pytest.mark.usefixtures("_mock_torch_env")
def test_train_model_pytorch_error(monkeypatch: object) -> object:
    """Initialize function test_train_model_pytorch_error.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """

    def mock_raise_error(*_args: object, **_kwargs: object) -> object:
        """Initialize function Exception.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.


        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(tr, "build_dataloader", Exception)
    train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1))


@pytest.mark.usefixtures("_mock_torch_env")
def test_train_model_pytorch_no_loader_fallback(monkeypatch: object) -> object:
    """Initialize function test_train_model_pytorch_no_loader_fallback.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Initialize function mock_build_dataloader.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.


        Returns:
            object: Description of return.

        """
        return {"loader": None}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1))


class MockModelObj:
    """Provide class docstring."""

    def to(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self

    def parameters(self) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return [1]

    def train(self) -> None:
        """Execute function."""

    def __call__(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return type("Out", (), {"view": lambda _self, *_a: _self, "size": lambda _self, *_a: 1, "logits": type("L", (), {"view": lambda _self, *_a: _self, "size": lambda _self, *_a: 1})()})()

    def view(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self

    def size(self, *_args: object, **_kwargs: object) -> int:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return 1


class MockModel:
    """Provide class docstring."""

    @classmethod
    def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockModelObj()


def test_train_model_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    m_train = __import__("gemma_4_sql.backends.pytorch.train", fromlist=[""])
    monkeypatch.setattr(m_train, "torch", MockTorch)
    monkeypatch.setattr(m_train, "optim", MockOptim)
    monkeypatch.setattr(m_train, "nn", MockNN)
    monkeypatch.setattr(m_train, "Gemma4ForCausalLM", MockModel)
    monkeypatch.setattr(m_train, "build_dataloader", lambda *_args, **_kwargs: {})
    res = m_train.train_model(TrainingConfig(action="sft", model_name="m", dataset="ds", epochs=1, learning_rate=0.1))
    assert res["status"] == "completed"
    monkeypatch.setattr(m_train, "build_dataloader", lambda *_args, **_kwargs: {"loader": [{"inputs": MockModelObj(), "targets": MockModelObj()}]})
    res = m_train.train_model(TrainingConfig(action="sft", model_name="m", dataset="ds", epochs=1, learning_rate=0.1))
    assert res["status"] == "completed"
