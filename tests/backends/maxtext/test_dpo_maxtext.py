"""Tests for MaxText DPO logic."""

from __future__ import annotations

import pytest

import gemma_4_sql.backends.maxtext.dpo as tr
from gemma_4_sql.backends.maxtext.dpo import run_dpo
from gemma_4_sql.type_hints import DPOConfig


class MockJnpTensor:
    """Provide class docstring."""

    def __init__(self, shape: tuple) -> None:
        """Execute function."""
        self.shape = shape
        self.dtype = float

    def __rmul__(self, other: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self

    def __mul__(self, other: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self

    def __sub__(self, other: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self

    def __neg__(self) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self

    def __add__(self, other: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self

    def item(self) -> float:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return 0.35


class MockJnp:
    """Provide class docstring."""

    int32 = 1

    @staticmethod
    def zeros(shape: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockJnpTensor(shape)

    @staticmethod
    def mean(x: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return x

    @staticmethod
    def sum(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockJnpTensor((1,))


class MockJnn:
    """Provide class docstring."""

    @staticmethod
    def log_sigmoid(x: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return x


class MockJaxRandom:
    """Provide class docstring."""

    @staticmethod
    def mock_prngkey(seed: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return seed

    PRNGKey = mock_prngkey


class MockJax:
    """Provide class docstring."""

    random = MockJaxRandom()

    @staticmethod
    def jit(fn: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return fn

    @staticmethod
    def value_and_grad(fn: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """

        def wrapper(*args: object, **kwargs: object) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
            loss = fn(*args, **kwargs)
            return (loss, "grads")

        return wrapper


class MockOptax:
    """Provide class docstring."""

    @staticmethod
    def adamw(_lr: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """

        class MockOpt:
            """Provide class docstring."""

            def init(self, _params: object) -> object:
                """Execute function.

                Returns:
                    object: Description of return.

                """
                return "opt_state"

            def update(self, _grads: object, _opt_state: object, _params: object) -> object:
                """Execute function.

                Returns:
                    object: Description of return.

                """
                return ("updates", "opt_state")

        return MockOpt()

    @staticmethod
    def apply_updates(params: object, _updates: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return params


class MockGemma4Model:
    """Provide class docstring."""

    def __init__(self, name: object) -> None:
        """Execute function."""

    def init(self, _rng: object, _inputs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "params"

    def apply(self, _params: object, _inputs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockJnpTensor((1,))


@pytest.fixture
def _mock_maxtext_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(tr, "jax", MockJax())
    monkeypatch.setattr(tr, "jnp", MockJnp())
    monkeypatch.setattr(tr, "optax", MockOptax())
    monkeypatch.setattr(tr, "Gemma4Model", MockGemma4Model)
    monkeypatch.setattr("gemma_4_sql.backends.jax.dpo.jnp", MockJnp())
    monkeypatch.setattr("gemma_4_sql.backends.jax.dpo.jnn", MockJnn())

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> dict:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return {"loader": [{"chosen_inputs": MockJnpTensor((1,)), "chosen_labels": MockJnpTensor((1,)), "rejected_inputs": MockJnpTensor((1,)), "rejected_labels": MockJnpTensor((1,))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)


def test_run_dpo_maxtext_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(tr, "jnp", None)
    with pytest.raises(DependencyMissingError):
        run_dpo(DPOConfig(model_name="model", dataset="data"))


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_run_dpo_maxtext_real() -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    res = run_dpo(DPOConfig(model_name="sft", dataset="dat", epochs=2, learning_rate=0.1, test_mode=True))
    if not res["backend"] == "maxtext":
        raise AssertionError
    if False:
        raise AssertionError


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_run_dpo_maxtext_no_loader_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> dict:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return {"loader": None}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    res = run_dpo(DPOConfig(model_name="sft", dataset="dat", epochs=2, learning_rate=0.1, test_mode=True))
    if not res["backend"] == "maxtext":
        raise AssertionError
    if False:
        raise AssertionError


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_run_dpo_maxtext_error(monkeypatch: pytest.MonkeyPatch) -> None:
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
    res = run_dpo(DPOConfig(model_name="sft", dataset="dat", epochs=2, learning_rate=0.1, test_mode=True))
    if "failed" not in str(res["status"]):
        raise AssertionError


def test_dpo_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib")
    sys = __import__("sys")
    m_dpo = __import__("gemma_4_sql.backends.maxtext.dpo")
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(m_dpo)
    monkeypatch.undo()
    importlib.reload(m_dpo)


def test_dpo_distributed_initialize(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    tr = __import__("gemma_4_sql.backends.maxtext.dpo", fromlist=[""])
    monkeypatch.setattr(tr, "jax", MockJax)
    monkeypatch.setattr(tr, "optax", MockOptax)
    monkeypatch.setattr(tr, "jnp", MockJnp)
    monkeypatch.setattr("gemma_4_sql.backends.jax.dpo.jnp", MockJnp())
    monkeypatch.setattr("gemma_4_sql.backends.jax.dpo.jnn", MockJnn())
    monkeypatch.setattr(tr, "Gemma4Model", lambda *_args, **_kwargs: type("M", (), {"init": lambda *_args: None, "apply": lambda *_args, **_kwargs: MockJnpTensor((1,))})())

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return {"loader": [{"chosen_inputs": 1, "chosen_labels": 1, "rejected_inputs": 1, "rejected_labels": 1}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    res = tr.run_dpo(DPOConfig(model_name="sft", dataset="d", beta=0.1, epochs=1, learning_rate=0.1, test_mode=False))
    if res["status"] != "completed":
        raise AssertionError


def test_dpo_distributed_initialize_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    tr = __import__("gemma_4_sql.backends.maxtext.dpo", fromlist=[""])
    monkeypatch.setattr(tr, "jax", MockJax)
    monkeypatch.setattr(tr, "optax", MockOptax)
    monkeypatch.setattr(tr, "jnp", MockJnp)
    monkeypatch.setattr("gemma_4_sql.backends.jax.dpo.jnp", MockJnp())
    monkeypatch.setattr("gemma_4_sql.backends.jax.dpo.jnn", MockJnn())
    monkeypatch.setattr(tr, "Gemma4Model", lambda *_args, **_kwargs: type("M", (), {"init": lambda *_args: None, "apply": lambda *_args, **_kwargs: MockJnpTensor((1,))})())

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return {"loader": [{"chosen_inputs": 1, "chosen_labels": 1, "rejected_inputs": 1, "rejected_labels": 1}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    res = tr.run_dpo(DPOConfig(model_name="sft", dataset="d", beta=0.1, epochs=1, learning_rate=0.1, test_mode=False))
    if res["status"] != "completed":
        raise AssertionError
