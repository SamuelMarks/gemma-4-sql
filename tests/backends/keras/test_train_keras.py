"""Tests for Keras Train module."""

from __future__ import annotations

import pytest

import gemma_4_sql.backends.keras.train as tr


class MockKerasModel:
    """Provide class docstring."""

    def __init__(self, vocab_size: int = 100) -> None:
        """Execute function."""
        self.vocab_size = vocab_size

    def __call__(self, x: object, *, _training: bool = False) -> object:
        """Execute function."""
        return x

    def compile(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return None

    def fit(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return type("History", (), {"history": {"loss": [0.5]}})()

    @classmethod
    def from_preset(cls, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        m = cls()
        m.preprocessor = type("PP", (), {"sequence_length": 512})()
        return m


class MockKeras:
    """Provide class docstring."""

    class MockOptimizers:
        """Provide class docstring."""

        @staticmethod
        def mock_adamw(*_args: object, **_kwargs: object) -> str:
            """Execute function."""
            return "opt"

        AdamW = mock_adamw

    optimizers = MockOptimizers

    class MockLosses:
        """Provide class docstring."""

        @staticmethod
        def mock_sparsecategoricalcrossentropy(*_args: object, **_kwargs: object) -> str:
            """Execute function."""
            return "loss"

        SparseCategoricalCrossentropy = mock_sparsecategoricalcrossentropy

    losses = MockLosses
    Model = MockKerasModel

    def mock_input(*_args: object, **_kwargs: object) -> str:
        """Execute function."""
        return "input"

    Input = mock_input

    class MockLayers:
        """Provide class docstring."""

        def mock_embedding(*_args: object, **_kwargs: object) -> object:
            """Execute function."""
            return lambda _x: "emb"

        Embedding = mock_embedding

        def mock_dense(*_args: object, **_kwargs: object) -> object:
            """Execute function."""
            return lambda _x: "out"

        Dense = mock_dense

    layers = MockLayers


class MockTf:
    """Provide class docstring."""

    @staticmethod
    def zeros(*_args: object, **_kwargs: object) -> str:
        """Execute function."""
        return "zeros"


@pytest.fixture
def _mock_keras_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(tr, "keras", MockKeras)
    monkeypatch.setattr(tr, "tf", MockTf)


pytestmark = pytest.mark.usefixtures("_mock_keras_env")


def test_train_model_keras_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    sys = __import__("sys")
    monkeypatch.setitem(sys.modules, "keras_nlp", type("MockKerasNLP", (), {}))
    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("MockModels", (), {"GemmaCausalLM": tr.keras.Model}))
    builtins = __import__("builtins")
    orig_import = builtins.__import__

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Execute function."""
        if name == "keras_nlp.models":
            return sys.modules["keras_nlp.models"]
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)
    monkeypatch.setattr(tr, "build_dataloader", lambda *_args, **_kwargs: {"loader": [1]})
    res = tr.train_model("sft", "mod", "dat", epochs=2)
    if res["status"] != "completed":
        raise AssertionError


def test_train_model_keras_missing() -> None:
    """Execute function."""
    tr.keras = None
    tr.tf = None
    res = tr.train_model("sft", "mod", "dat", epochs=2)
    if res["status"] != "mocked_missing_keras":
        raise AssertionError


def test_train_model_keras_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""

    def mock_raise_error(*_args: object, **_kwargs: object) -> object:
        """Execute function."""
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(tr, "build_dataloader", Exception)
    res = tr.train_model("sft", "mod", "dat", epochs=2)
    if "failed" not in res["status"]:
        raise AssertionError


def test_train_model_keras_no_loader_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(tr, "build_dataloader", lambda *_args, **_kwargs: {"loader": None})
    res = tr.train_model("sft", "mod", "dat", test_mode=True, epochs=2)
    if res["status"] != "completed":
        raise AssertionError


def test_train_keras_real_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(tr, "build_dataloader", lambda *_args, **_kwargs: {"loader": None})
    res = tr.train_model("sft", "model", "ds", test_mode=True, epochs=1)
    if res["status"] != "completed":
        raise AssertionError


def test_train_keras_real_import_with_loader_iter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(tr, "build_dataloader", lambda *_args, **_kwargs: {"loader": [1, 2]})
    res = tr.train_model("sft", "model", "ds", test_mode=True, epochs=1)
    if res["status"] != "completed":
        raise AssertionError
