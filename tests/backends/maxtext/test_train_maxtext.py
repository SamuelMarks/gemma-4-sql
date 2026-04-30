"""Tests for MaxText training pipeline."""

import pytest

import gemma_4_sql.backends.maxtext.train as tr
from gemma_4_sql.backends.maxtext.train import train_model


class MockJnpTensor:
    def __init__(self, shape):
        self.shape = shape

    def item(self):
        return 0.35


class MockJnp:
    int32 = 1

    @staticmethod
    def zeros(shape, dtype=None):
        return MockJnpTensor(shape)

    @staticmethod
    def mean(x):
        return x


class MockJaxRandom:
    @staticmethod
    def PRNGKey(seed):
        return seed


class MockJax:
    random = MockJaxRandom()

    @staticmethod
    def jit(fn):
        return fn

    @staticmethod
    def value_and_grad(fn):
        def wrapper(*args, **kwargs):
            _ = fn(*args, **kwargs)
            return MockJnpTensor((1,)), "grads"

        return wrapper


class MockOptax:
    @staticmethod
    def adamw(lr):
        class MockOpt:
            def init(self, params):
                return "opt_state"

            def update(self, grads, opt_state, params):
                return "updates", "opt_state"

        return MockOpt()

    @staticmethod
    def softmax_cross_entropy_with_integer_labels(logits, labels):
        return MockJnpTensor((1,))

    @staticmethod
    def apply_updates(params, updates):
        return params


class MockGemma4Model:
    def __init__(self, name):
        pass

    def init(self, rng, inputs):
        return "params"

    def apply(self, params, inputs):
        return MockJnpTensor((1,))


@pytest.fixture
def mock_maxtext_env(monkeypatch):
    monkeypatch.setattr(tr, "jax", MockJax())
    monkeypatch.setattr(tr, "jnp", MockJnp())
    monkeypatch.setattr(tr, "optax", MockOptax())
    monkeypatch.setattr(tr, "Gemma4Model", MockGemma4Model)

    def mock_build_dataloader(*args, **kwargs):
        return {
            "loader": [{"inputs": MockJnpTensor((1,)), "targets": MockJnpTensor((1,))}]
        }

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)


def test_train_model_maxtext_real(mock_maxtext_env):
    res = train_model("sft", "mod", "dat", 2, 0.1)
    assert res["status"] == "completed"
    assert res["final_loss"] == 0.35
    assert res["backend"] == "maxtext"


def test_train_model_maxtext_missing():
    orig_jax = tr.jax
    tr.jax = None
    res = train_model("sft", "mod", "dat", 2, 0.1)
    assert res["status"] == "mocked_missing_maxtext"
    tr.jax = orig_jax


def test_train_model_maxtext_error(mock_maxtext_env, monkeypatch):
    def raise_error(*args, **kwargs):
        raise ValueError("err")

    monkeypatch.setattr(tr, "build_dataloader", raise_error)
    res = train_model("sft", "mod", "dat", 2, 0.1)
    assert "failed: err" in res["status"]


def test_train_model_maxtext_no_loader_fallback(mock_maxtext_env, monkeypatch):
    def mock_build_dataloader(*args, **kwargs):
        return {"loader": None}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    res = train_model("sft", "mod", "dat", 2, 0.1)
    assert res["status"] == "completed"
    assert res["final_loss"] == 0.35
