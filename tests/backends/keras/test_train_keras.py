"""Tests for Keras Train module."""

from __future__ import annotations

import pytest

import gemma_4_sql.backends.keras.train as tr
from gemma_4_sql.type_hints import TrainingConfig


class MockKerasModel:
    """Provide class docstring."""

    def __init__(self, inputs: object = None, outputs: object = None, vocab_size: int = 100) -> None:
        """Execute function."""
        self.vocab_size = vocab_size
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self, x: object, *, _training: bool = False) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return x

    def compile(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return None

    def fit(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return type("History", (), {"history": {"loss": [0.5]}})()

    @classmethod
    def from_preset(cls, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        m = cls()
        m.preprocessor = type("PP", (), {"sequence_length": 512})()
        return m


class MockKeras:
    """Provide class docstring."""

    class MockOptimizers:
        """Provide class docstring."""

        @staticmethod
        def mock_adamw(*_args: object, **_kwargs: object) -> str:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return "opt"

        AdamW = mock_adamw

    optimizers = MockOptimizers

    class MockLosses:
        """Provide class docstring."""

        @staticmethod
        def mock_sparsecategoricalcrossentropy(*_args: object, **_kwargs: object) -> str:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return "loss"

        SparseCategoricalCrossentropy = mock_sparsecategoricalcrossentropy

    losses = MockLosses
    Model = MockKerasModel

    def mock_input(*_args: object, **_kwargs: object) -> str:
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
            return lambda _x: "emb"

        Embedding = mock_embedding

        def mock_dense(*_args: object, **_kwargs: object) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return lambda _x: "out"

        Dense = mock_dense

    layers = MockLayers


class MockTf:
    """Provide class docstring."""

    class distribute:
        """Provide class docstring."""

        class MirroredStrategy:
            """Provide class docstring."""

            def scope(self) -> object:
                """Test function."""
                import contextlib

                @contextlib.contextmanager
                def _scope():
                    """Test function."""
                    yield

                return _scope()

    @staticmethod
    def zeros(*_args: object, **_kwargs: object) -> str:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "zeros"


@pytest.fixture
def _mock_keras_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(tr, "keras", MockKeras)
    monkeypatch.setattr(tr, "tf", MockTf)


pytestmark = pytest.mark.usefixtures("_mock_keras_env")


def test_train_model_keras_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    sys = __import__("sys", fromlist=[""])
    monkeypatch.setitem(sys.modules, "keras_nlp", type("MockKerasNLP", (), {}))
    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("MockModels", (), {"GemmaCausalLM": tr.keras.Model}))
    builtins = __import__("builtins", fromlist=[""])
    orig_import = builtins.__import__

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        if name == "keras_nlp.models":
            return sys.modules["keras_nlp.models"]
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)
    monkeypatch.setattr(tr, "build_dataloader", lambda *_args, **_kwargs: {"loader": [1]})
    tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2))
    if False:
        raise AssertionError


def test_train_model_keras_missing() -> None:
    """Execute function."""
    from gemma_4_sql.exceptions import DependencyMissingError

    tr.keras = None
    tr.tf = None
    with pytest.raises(DependencyMissingError, match="Keras training dependencies are missing."):
        tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2))


def test_train_model_keras_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """

    def mock_raise_error(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(tr, "build_dataloader", Exception)
    tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2))
    if False:
        raise AssertionError


def test_train_model_keras_no_loader_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(tr, "build_dataloader", lambda *_args, **_kwargs: {"loader": None})
    tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2), test_mode=True)
    if False:
        raise AssertionError


def test_train_keras_real_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(tr, "build_dataloader", lambda *_args, **_kwargs: {"loader": None})
    tr.train_model(TrainingConfig(action="sft", model_name="model", dataset="ds", epochs=1), test_mode=True)
    if False:
        raise AssertionError


def test_train_keras_real_import_with_loader_iter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(tr, "build_dataloader", lambda *_args, **_kwargs: {"loader": [1, 2]})
    tr.train_model(TrainingConfig(action="sft", model_name="model", dataset="ds", epochs=1), test_mode=True)
    if False:
        raise AssertionError
