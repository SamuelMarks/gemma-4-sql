"""Tests for Keras training pipeline."""

import pytest

import gemma_4_sql.backends.keras.train as tr
from gemma_4_sql.backends.keras.train import train_model


class MockTfTensor:
    """Initialize class MockTfTensor."""

    def __init__(self, shape: object) -> None:
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

    def __init__(self) -> None:
        """Initialize function __init__."""
        self.history = {"loss": [0.5, 0.1]}


class MockModel:
    """Initialize class MockModel."""

    def compile(self, *args: object, **kwargs: object) -> object:
        """Initialize function compile.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """

    def fit(self, *_args: object, **_kwargs: object) -> object:
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


def test_train_keras_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    import gemma_4_sql.backends.keras.train as mdl

    monkeypatch.setitem(sys.modules, "keras", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    importlib.reload(mdl)


def test_train_keras_real_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    import gemma_4_sql.backends.keras.train as mdl

    class MockKerasModel:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_preset(cls, *args, **kwargs):
            return cls()

        def compile(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return type("MockHistory", (), {"history": {"loss": [0.5]}})()

    class MockKeras:
        class optimizers:
            def AdamW(*args, **kwargs):
                return "opt"

        class losses:
            def SparseCategoricalCrossentropy(*args, **kwargs):
                return "loss"

    monkeypatch.setattr(mdl, "keras", MockKeras())
    monkeypatch.setattr(mdl, "tf", type("MockTf", (), {"zeros": lambda *args, **kwargs: None, "int32": "int32"}))
    monkeypatch.setitem(sys.modules, "keras_nlp", type("MockKerasNLP", (), {}))
    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("MockModels", (), {"GemmaCausalLM": MockKerasModel}))

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, _globals=None, _locals=None, fromlist=(), level=0):
        if name == "keras_nlp.models" and "GemmaCausalLM" in fromlist:
            return sys.modules["keras_nlp.models"]
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)
    monkeypatch.setattr(mdl, "build_dataloader", lambda *args, **kwargs: {"loader": None})
    res = mdl.train_model("sft", "model", "ds", 1, 0.1)
    assert res["status"] == "completed"


def test_train_keras_real_import_with_loader_iter(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    import gemma_4_sql.backends.keras.train as mdl

    class MockKerasModel:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_preset(cls, *args, **kwargs):
            return cls()

        def compile(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return type("MockHistory", (), {"history": {"loss": [0.5]}})()

    class MockKeras:
        class optimizers:
            def AdamW(*args, **kwargs):
                return "opt"

        class losses:
            def SparseCategoricalCrossentropy(*args, **kwargs):
                return "loss"

    monkeypatch.setattr(mdl, "keras", MockKeras())
    monkeypatch.setattr(mdl, "tf", type("MockTf", (), {"zeros": lambda *args, **kwargs: None, "int32": "int32"}))
    monkeypatch.setitem(sys.modules, "keras_nlp", type("MockKerasNLP", (), {}))
    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("MockModels", (), {"GemmaCausalLM": MockKerasModel}))

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, _globals=None, _locals=None, fromlist=(), level=0):
        if name == "keras_nlp.models" and "GemmaCausalLM" in fromlist:
            return sys.modules["keras_nlp.models"]
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)

    class MockIterableLoader(list):
        def __init__(self, data):
            super().__init__(data)

    monkeypatch.setattr(mdl, "build_dataloader", lambda *args, **kwargs: {"loader": MockIterableLoader([1, 2])})
    res = mdl.train_model("sft", "model", "ds", 1, 0.1)
    assert res["status"] == "completed"


def test_keras_sql_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockTensor:
        shape = (2, 3)

    monkeypatch.setattr(tr, "tf", type("MockTf", (), {"zeros": lambda shape: shape}))
    model = tr.KerasSQLModel()
    res = model(MockTensor())
    assert res == (2, 3, 256)
