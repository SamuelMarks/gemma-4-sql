"""Tests for JAX training pipeline."""

import sys
from unittest import mock

import pytest

import gemma_4_sql.backends.jax.train as tr
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import TrainerState, TrainingConfig


class MockJnpTensor:
    def __init__(self, val=0.35) -> None:
        self.val = val
        self.shape = (1,)

    def item(self) -> float:
        return self.val

    def __float__(self):
        return float(self.val)


@pytest.fixture
def _mock_jax_train_env(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockJax:
        class sharding:
            class Mesh:
                def __init__(self, *args):
                    pass

            class NamedSharding:
                def __init__(self, *args):
                    pass

            class PartitionSpec:
                def __init__(self, *args):
                    pass

        @staticmethod
        def devices():
            return ["cpu"]

        @staticmethod
        def device_put(x, params=None):
            return x

    class MockJnp:
        @staticmethod
        def mean(x):
            return MockJnpTensor(0.5)

    class MockOptax:
        @staticmethod
        def warmup_cosine_decay_schedule(**kwargs):
            return "schedule"

        @staticmethod
        def adamw(schedule):
            return "adamw"

        @staticmethod
        def clip_by_global_norm(clip):
            return "clip"

        @staticmethod
        def chain(*args):
            return "chain"

        @staticmethod
        def softmax_cross_entropy_with_integer_labels(logits, targets):
            return MockJnpTensor(0.5)

    class MockNNX:
        class Rngs:
            def __init__(self, seed):
                pass

        class Optimizer:
            def __init__(self, model, tx):
                pass

            def update(self, grads):
                pass

        @staticmethod
        def jit(fn):
            return fn

        @staticmethod
        def value_and_grad(fn):
            def wrapper(*args, **kwargs):
                return MockJnpTensor(0.123), "mock_grads"

            return wrapper

    class MockGemma4Config:
        @staticmethod
        def gemma4_e2b():
            return "mock_config"

    class MockGemma4ForCausalLM:
        def __init__(self, config, rngs):
            pass

        def __call__(self, inputs):
            return MockJnpTensor(0.99)

    monkeypatch.setattr(tr, "jax", MockJax())
    monkeypatch.setattr(tr, "jnp", MockJnp())
    monkeypatch.setattr(tr, "optax", MockOptax())
    monkeypatch.setattr(tr, "nnx", MockNNX())
    monkeypatch.setattr(tr, "Gemma4Config", MockGemma4Config)
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", MockGemma4ForCausalLM)

    def mock_build_dataloader(*args, **kwargs):
        return {"loader": [{"inputs": [1], "targets": [1]}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)


@pytest.mark.usefixtures("_mock_jax_train_env")
def test_train_model_jax_real() -> None:
    res = tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=1, learning_rate=0.1))
    assert res["backend"] == "jax"
    assert res["status"] == "completed"
    assert res["final_loss"] == pytest.approx(0.123)


@pytest.mark.usefixtures("_mock_jax_train_env")
def test_train_model_jax_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_raise_error(*args, **kwargs):
        raise ValueError("mock build dataloader error")

    monkeypatch.setattr(tr, "build_dataloader", mock_raise_error)
    res = tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=1, learning_rate=0.1))
    assert res["status"] == "failed: mock build dataloader error"


def test_train_model_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tr, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing for training\."):
        tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=1, learning_rate=0.1))


@pytest.mark.usefixtures("_mock_jax_train_env")
def test_execute_train_no_loader_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_build_dataloader(*args, **kwargs):
        return {"loader": None}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    with pytest.raises(ValueError, match="Invalid dataloader"):
        tr._execute_train("dat", 1, 0.1)


@pytest.mark.usefixtures("_mock_jax_train_env")
def test_loss_fn() -> None:
    def model(x):
        return MockJnpTensor(0.99)

    batch = {"inputs": [1], "targets": [1]}
    loss = tr._loss_fn(model, batch)
    assert loss.val == pytest.approx(0.5)


@pytest.mark.usefixtures("_mock_jax_train_env")
def test_get_train_step_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    step_fn = tr._get_train_step_fn()

    class MockOpt:
        def update(self, grads):
            self.grads = grads

    opt = MockOpt()
    loss = step_fn(None, opt, {})
    assert loss.val == pytest.approx(0.123)
    assert opt.grads == "mock_grads"

    # Test fallback branches when nnx missing methods
    class MockNNXNoGrad:
        @staticmethod
        def jit(fn):
            return fn

    monkeypatch.setattr(tr, "nnx", MockNNXNoGrad())
    step_fn_no_grad = tr._get_train_step_fn()
    loss2 = step_fn_no_grad(None, None, {})
    assert loss2 == 0.0

    monkeypatch.setattr(tr, "nnx", None)
    step_fn_no_nnx = tr._get_train_step_fn()
    loss3 = step_fn_no_nnx(None, None, {})
    assert loss3 == 0.0


@pytest.mark.usefixtures("_mock_jax_train_env")
def test_run_training_epochs() -> None:
    def mock_train_step(model, opt, batch):
        return MockJnpTensor(0.42)

    state = TrainerState(
        dataloader=[{"inputs": [1], "targets": [1]}],
        epochs=1,
        policy_model=None,
        optimizer=None,
        train_step=mock_train_step,
        params=None,
    )
    final_loss = tr._run_training_epochs(state)
    assert final_loss == pytest.approx(0.42)


def test_train_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    with mock.patch.dict(sys.modules, {"jax": None, "flax": None}):
        importlib.reload(tr)
        assert tr.jax is None
        assert tr.nnx is None

    importlib.reload(tr)


def test_execute_train_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tr, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing for training\."):
        tr._execute_train("dat", 1, 0.1)
