"""Tests for MaxText training pipeline."""

import typing

import pytest

import gemma_4_sql.backends.maxtext.train as tr
from gemma_4_sql.backends.maxtext.train import train_model


class MockJnpTensor:
    """Initialize class MockJnpTensor."""

    def __init__(self: typing.Any, shape: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        shape: Description of shape.

        """
        self.shape = shape

    def item(self: typing.Any) -> object:
        """Initialize function item."""
        return 0.35


class MockJnp:
    """Initialize class MockJnp."""

    int32 = 1

    @staticmethod
    def zeros(shape: object, _dtype: object = None) -> object:
        """Initialize function zeros.

        Args:
        ----
        shape: Description of shape.
        dtype: Description of dtype.

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
    def prngkey(seed: object) -> object:
        """Initialize function prngkey.

        Args:
        ----
        seed: Description of seed.

        """
        return seed


class MockJax:
    """Initialize class MockJax."""

    random = MockJaxRandom()

    @staticmethod
    def jit(fn: object) -> object:
        """Initialize function jit.

        Args:
        ----
        fn: Description of fn.

        """
        return fn

    @staticmethod
    def value_and_grad(fn: object) -> object:
        """Initialize function value_and_grad.

        Args:
        ----
        fn: Description of fn.

        """

        def wrapper(*args: object, **kwargs: object) -> object:
            """Initialize function wrapper.

            Args:
            ----
            args: Description of args.
            kwargs: Description of kwargs.

            """
            _ = fn(*args, **kwargs)  # type: ignore[operator]
            return (MockJnpTensor((1,)), "grads")

        return wrapper


class MockOptax:
    """Initialize class MockOptax."""

    @staticmethod
    def adamw(_lr: object) -> object:
        """Initialize function adamw.

        Args:
        ----
        lr: Description of lr.

        """

        class MockOpt:
            """Initialize class MockOpt."""

            def init(self: typing.Any, _params: object) -> object:
                """Initialize function init.

                Args:
                ----
                params: Description of params.

                """
                return "opt_state"

            def update(self: typing.Any, _grads: object, _opt_state: object, _params: object) -> object:
                """Initialize function update.

                Args:
                ----
                grads: Description of grads.
                opt_state: Description of opt_state.
                params: Description of params.

                """
                return ("updates", "opt_state")

        return MockOpt()

    @staticmethod
    def softmax_cross_entropy_with_integer_labels(_logits: object, _labels: object) -> object:
        """Initialize function softmax_cross_entropy_with_integer_labels.

        Args:
        ----
        logits: Description of logits.
        labels: Description of labels.

        """
        return MockJnpTensor((1,))

    @staticmethod
    def apply_updates(params: object, _updates: object) -> object:
        """Initialize function apply_updates.

        Args:
        ----
        params: Description of params.
        updates: Description of updates.

        """
        return params


class MockGemma4Model:
    """Initialize class MockGemma4Model."""

    def __init__(self: typing.Any, name: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        name: Description of name.

        """

    def init(self: typing.Any, _rng: object, _inputs: object) -> object:
        """Initialize function init.

        Args:
        ----
        rng: Description of rng.
        inputs: Description of inputs.

        """
        return "params"

    def apply(self: typing.Any, _params: object, _inputs: object) -> object:
        """Initialize function apply.

        Args:
        ----
        params: Description of params.
        inputs: Description of inputs.

        """
        return MockJnpTensor((1,))


@pytest.fixture
def _mock_maxtext_env(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function mock_maxtext_env.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(tr, "jax", MockJax())  # type: ignore[attr-defined]
    monkeypatch.setattr(tr, "jnp", MockJnp())  # type: ignore[attr-defined]
    monkeypatch.setattr(tr, "optax", MockOptax())  # type: ignore[attr-defined]
    monkeypatch.setattr(tr, "Gemma4Model", MockGemma4Model)  # type: ignore[attr-defined]

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Initialize function mock_build_dataloader.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        return {"loader": [{"inputs": MockJnpTensor((1,)), "targets": MockJnpTensor((1,))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)  # type: ignore[attr-defined]


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_train_model_maxtext_real() -> object:  # type: ignore[return]
    """Initialize function test_train_model_maxtext_real.

    Args:
    ----
    mock_maxtext_env: Description of mock_maxtext_env.

    """
    res = train_model("sft", "mod", "dat", 2, 0.1)
    if not res["backend"] == "maxtext":
        raise AssertionError


def test_train_model_maxtext_missing() -> object:  # type: ignore[return]
    """Initialize function test_train_model_maxtext_missing."""
    orig_jax = tr.jax  # type: ignore[attr-defined]
    tr.jax = None  # type: ignore[attr-defined]
    res = train_model("sft", "mod", "dat", 2, 0.1)
    if not res["status"] == "mocked_missing_maxtext":
        raise AssertionError
    tr.jax = orig_jax  # type: ignore[attr-defined]


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_train_model_maxtext_error(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function test_train_model_maxtext_error.

    Args:
    ----
    mock_maxtext_env: Description of mock_maxtext_env.
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


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_train_model_maxtext_no_loader_fallback(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function test_train_model_maxtext_no_loader_fallback.

    Args:
    ----
    mock_maxtext_env: Description of mock_maxtext_env.
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
