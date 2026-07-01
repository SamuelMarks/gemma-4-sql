"""Tests for MaxText DPO logic."""

from __future__ import annotations

import pytest

import gemma_4_sql.backends.maxtext.dpo as tr
from gemma_4_sql.backends.maxtext.dpo import run_dpo


class MockJnpTensor:
    def __init__(self, shape: tuple) -> None:
        self.shape = shape
        self.dtype = float

    def __rmul__(self, other: object) -> object:
        return self

    def __mul__(self, other: object) -> object:
        return self

    def __sub__(self, other: object) -> object:
        return self

    def __neg__(self) -> object:
        return self

    def __add__(self, other: object) -> object:
        return self

    def item(self) -> float:
        return 0.35


class MockJnp:
    int32 = 1

    @staticmethod
    def zeros(shape: object, **kwargs: object) -> object:
        return MockJnpTensor(shape)  # type: ignore[arg-type]

    @staticmethod
    def mean(x: object) -> object:
        return x

    @staticmethod
    def sum(*args: object, **kwargs: object) -> object:
        return MockJnpTensor((1,))


class MockJnn:
    @staticmethod
    def log_sigmoid(x: object) -> object:
        return x


class MockJaxRandom:
    @staticmethod
    def PRNGKey(seed: object) -> object:
        return seed


class MockJax:
    random = MockJaxRandom()

    @staticmethod
    def jit(fn: object) -> object:
        return fn

    @staticmethod
    def value_and_grad(fn: object) -> object:
        def wrapper(*args: object, **kwargs: object) -> object:
            _ = fn(*args, **kwargs)  # type: ignore[operator]
            return (MockJnpTensor((1,)), "grads")

        return wrapper


class MockOptax:
    @staticmethod
    def adamw(_lr: object) -> object:
        class MockOpt:
            def init(self, _params: object) -> object:
                return "opt_state"

            def update(self, _grads: object, _opt_state: object, _params: object) -> object:
                return ("updates", "opt_state")

        return MockOpt()

    @staticmethod
    def apply_updates(params: object, _updates: object) -> object:
        return params


class MockGemma4Model:
    def __init__(self, name: object) -> None:
        pass

    def init(self, _rng: object, _inputs: object) -> object:
        return "params"

    def apply(self, _params: object, _inputs: object) -> object:
        return MockJnpTensor((1,))


@pytest.fixture
def _mock_maxtext_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tr, "jax", MockJax())
    monkeypatch.setattr(tr, "jnp", MockJnp())
    monkeypatch.setattr(tr, "optax", MockOptax())
    monkeypatch.setattr(tr, "Gemma4Model", MockGemma4Model)
    monkeypatch.setattr("gemma_4_sql.backends.jax.dpo.jnp", MockJnp())
    monkeypatch.setattr("gemma_4_sql.backends.jax.dpo.jnn", MockJnn())

    def mock_build_dataloader(*args: object, **kwargs: object) -> dict:
        return {"loader": [{"chosen_inputs": MockJnpTensor((1,)), "chosen_labels": MockJnpTensor((1,)), "rejected_inputs": MockJnpTensor((1,)), "rejected_labels": MockJnpTensor((1,))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)


def test_run_dpo_maxtext_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tr, "jnp", None)
    res = run_dpo("model", "data")
    if not res["status"] == "mocked_missing_maxtext":
        raise AssertionError


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_run_dpo_maxtext_real() -> None:
    res = run_dpo("sft", "dat", epochs=2, learning_rate=0.1, test_mode=True)
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["status"] == "completed":
        raise AssertionError


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_run_dpo_maxtext_no_loader_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_build_dataloader(*args: object, **kwargs: object) -> dict:
        return {"loader": None}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)

    res = run_dpo("sft", "dat", epochs=2, learning_rate=0.1, test_mode=True)
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["status"] == "completed":
        raise AssertionError


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_run_dpo_maxtext_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_error(*args: object, **kwargs: object) -> object:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(tr, "build_dataloader", raise_error)

    res = run_dpo("sft", "dat", epochs=2, learning_rate=0.1, test_mode=True)
    if "failed" not in str(res["status"]):
        raise AssertionError


def test_dpo_imports_fail(monkeypatch: pytest.MonkeyPatch):
    import importlib
    import sys

    import gemma_4_sql.backends.maxtext.dpo as m_dpo

    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(m_dpo)
    monkeypatch.undo()
    importlib.reload(m_dpo)


def test_dpo_distributed_initialize(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.maxtext.dpo as tr

    class MockJax:
        jit = lambda x: x
        value_and_grad = lambda x, **kw: lambda *a, **kw: (type("L", (), {"item": lambda self: 1.0})(), 1)

        class random:
            PRNGKey = lambda x: x
            split = lambda x: (x, x)

        class distributed:
            @staticmethod
            def initialize():
                pass

    class MockOptax:
        apply_updates = lambda *args, **kwargs: None
        adamw = lambda *args, **kwargs: type("Opt", (), {"init": lambda self, x: None, "update": lambda *a, **kw: (None, None)})()

    class MockJnp:
        ones = lambda *args, **kwargs: 1
        zeros = lambda *args, **kwargs: 1
        int32 = 1
        mean = lambda *args, **kwargs: 1

    monkeypatch.setattr(tr, "jax", MockJax)
    monkeypatch.setattr(tr, "optax", MockOptax)
    monkeypatch.setattr(tr, "jnp", MockJnp)
    monkeypatch.setattr(tr, "Gemma4Model", lambda *args, **kwargs: type("M", (), {"init": lambda *args: None, "apply": lambda *args, **kwargs: None})())
    res = tr.run_dpo("sft", "d", 1, 0.1, 0.1, test_mode=False)
    assert res["status"] == "completed"


def test_dpo_distributed_initialize_fail(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.maxtext.dpo as tr

    class MockJax:
        jit = lambda x: x
        value_and_grad = lambda x, **kw: lambda *a, **kw: (type("L", (), {"item": lambda self: 1.0})(), 1)

        class random:
            PRNGKey = lambda x: x
            split = lambda x: (x, x)

        class distributed:
            @staticmethod
            def initialize():
                msg = "err"
                raise RuntimeError(msg)

    class MockOptax:
        apply_updates = lambda *args, **kwargs: None
        adamw = lambda *args, **kwargs: type("Opt", (), {"init": lambda self, x: None, "update": lambda *a, **kw: (None, None)})()

    class MockJnp:
        ones = lambda *args, **kwargs: 1
        zeros = lambda *args, **kwargs: 1
        int32 = 1
        mean = lambda *args, **kwargs: 1

    monkeypatch.setattr(tr, "jax", MockJax)
    monkeypatch.setattr(tr, "optax", MockOptax)
    monkeypatch.setattr(tr, "jnp", MockJnp)
    monkeypatch.setattr(tr, "Gemma4Model", lambda *args, **kwargs: type("M", (), {"init": lambda *args: None, "apply": lambda *args, **kwargs: None})())
    res = tr.run_dpo("sft", "d", 1, 0.1, 0.1, test_mode=False)
    assert res["status"] == "completed"
