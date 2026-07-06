"""Tests for MaxText training pipeline."""

import pytest

import gemma_4_sql.backends.maxtext.train as tr
from gemma_4_sql.backends.maxtext.train import train_model
from gemma_4_sql.type_hints import TrainingConfig


class MockJnpTensor:
    """Initialize class MockJnpTensor."""

    def __init__(self, shape: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        shape: Description of shape.

        """
        self.shape = shape

    def item(self) -> object:
        """Initialize function item.

        Returns:
            object: Description of return.

        """
        return 0.35


class MockJnp:
    """Initialize class MockJnp."""

    int32 = 1

    @staticmethod
    def zeros(shape: object, **_kwargs: object) -> object:
        """Initialize function zeros.

        Args:
        ----
        shape: Description of shape.
        dtype: Description of dtype.
        **kwargs: Description of kwargs.


        Returns:
            object: Description of return.

        """
        return MockJnpTensor(shape)

    @staticmethod
    def mean(x: object) -> object:
        """Initialize function mean.

        Args:
        ----
        x: Description of x.

        """


class MockJaxRandom:
    """Initialize class MockJaxRandom."""

    @staticmethod
    def mock_prngkey(seed: object) -> object:
        """Initialize function prngkey.

        Args:
        ----
        seed: Description of seed.


        Returns:
            object: Description of return.

        """
        return seed

    PRNGKey = mock_prngkey


class MockJax:
    """Initialize class MockJax."""

    random = MockJaxRandom()

    @staticmethod
    def jit(fn: object) -> object:
        """Initialize function jit.

        Args:
        ----
        fn: Description of fn.


        Returns:
            object: Description of return.

        """
        return fn

    @staticmethod
    def value_and_grad(fn: object) -> object:
        """Initialize function value_and_grad.

        Args:
        ----
        fn: Description of fn.


        Returns:
            object: Description of return.

        """

        def wrapper(*args: object, **kwargs: object) -> object:
            """Initialize function wrapper.

            Args:
            ----
            args: Description of args.
            kwargs: Description of kwargs.


            Returns:
                object: Description of return.

            """
            _ = fn(*args, **kwargs)
            return (MockJnpTensor((1,)), "grads")

        return wrapper


class MockOptax:
    """Initialize class MockOptax."""

    @staticmethod
    def adamw(_lr: object) -> object:
        """Initialize function adamw.

        Returns:
            object: Description of return.

        """

        class MockOpt:
            """Initialize class MockOpt."""

            def init(self, _params: object) -> object:
                """Initialize function init.

                Returns:
                    object: Description of return.

                """
                return "opt_state"

            def update(self, _grads: object, _opt_state: object, _params: object) -> object:
                """Initialize function update.

                Returns:
                    object: Description of return.

                """
                return ("updates", "opt_state")

        return MockOpt()

    @staticmethod
    def softmax_cross_entropy_with_integer_labels(_logits: object, _labels: object) -> object:
        """Initialize function softmax_cross_entropy_with_integer_labels.

        Returns:
            object: Description of return.

        """
        return MockJnpTensor((1,))

    @staticmethod
    def apply_updates(params: object, _updates: object) -> object:
        """Initialize function apply_updates.

        Args:
        ----
        params: Description of params.


        Returns:
            object: Description of return.

        """
        return params


class MockGemma4Model:
    """Initialize class MockGemma4Model."""

    def __init__(self, name: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        name: Description of name.

        """

    def init(self, _rng: object, _inputs: object) -> object:
        """Initialize function init.

        Returns:
            object: Description of return.

        """
        return "params"

    def apply(self, _params: object, _inputs: object) -> object:
        """Initialize function apply.

        Returns:
            object: Description of return.

        """
        return MockJnpTensor((1,))


@pytest.fixture
def _mock_maxtext_env(monkeypatch: object) -> object:
    """Initialize function mock_maxtext_env.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(tr, "jax", MockJax())
    monkeypatch.setattr(tr, "jnp", MockJnp())
    monkeypatch.setattr(tr, "optax", MockOptax())
    monkeypatch.setattr(tr, "Gemma4Model", MockGemma4Model)

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Initialize function mock_build_dataloader.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.


        Returns:
            object: Description of return.

        """
        return {"loader": [{"inputs": MockJnpTensor((1,)), "targets": MockJnpTensor((1,))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_train_model_maxtext_real() -> object:
    """Initialize function test_train_model_maxtext_real.

    Raises:
        AssertionError: Description.

    """
    res = train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1))
    if not res["backend"] == "maxtext":
        raise AssertionError


def test_train_model_maxtext_missing() -> object:
    """Initialize function test_train_model_maxtext_missing.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    orig_jax = tr.jax
    tr.jax = None
    with pytest.raises(DependencyMissingError):
        train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1))
    tr.jax = orig_jax


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_train_model_maxtext_error(monkeypatch: object) -> object:
    """Initialize function test_train_model_maxtext_error.

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


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_train_model_maxtext_no_loader_fallback(monkeypatch: object) -> object:
    """Initialize function test_train_model_maxtext_no_loader_fallback.

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


def test_train_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    m_train = __import__("gemma_4_sql.backends.maxtext.train", fromlist=[""])
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(m_train)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "maxtext.train", None)
    importlib.reload(m_train)
    monkeypatch.undo()
    importlib.reload(m_train)


class MockMaxTextTrain:
    """Provide class docstring."""

    @staticmethod
    def main(*args: object, **kwargs: object) -> None:
        """Execute function."""


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_train_model_maxtext_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    m_train = __import__("gemma_4_sql.backends.maxtext.train", fromlist=[""])
    monkeypatch.setattr(m_train, "maxtext_train", MockMaxTextTrain())
    res = m_train.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1, extra_kwargs={"test_mode": False}))
    if res["status"] != "completed":
        raise AssertionError


def test_train_imports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    m_train = __import__("gemma_4_sql.backends.maxtext.train", fromlist=[""])
    monkeypatch.setitem(sys.modules, "maxtext.train", type("M", (), {})())
    monkeypatch.setitem(sys.modules, "maxtext", type("M", (), {})())
    builtins = __import__("builtins", fromlist=[""])
    orig_import = builtins.__import__

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        if name == "maxtext.models.gemma4" and "Gemma4Model" in fromlist:
            return type("M", (), {"Gemma4Model": "mocked_gemma4"})
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)
    importlib.reload(m_train)
    monkeypatch.undo()
    importlib.reload(m_train)
