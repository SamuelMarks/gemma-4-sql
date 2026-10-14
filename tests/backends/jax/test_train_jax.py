"""Tests for JAX training pipeline."""

from tests.conftest import requires_jax

pytestmark = requires_jax

import pytest

import gemma_4_sql.backends.jax.train as tr
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import TrainerState, TrainingConfig


class MockJnpTensor:
    """Verify behavior of MockJnpTensor."""

    def __init__(self, val=0.35) -> None:
        """Verify behavior of __init__."""
        self.val = val
        self.shape = (1,)

    def item(self) -> float:
        """Verify behavior of item."""
        return self.val

    def __float__(self):
        """Verify behavior of __float__."""
        return float(self.val)


@pytest.fixture
def _mock_jax_train_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify behavior of _mock_jax_train_env."""

    class MockJax:
        """Verify behavior of MockJax."""

        class sharding:
            """Verify behavior of sharding."""

            class Mesh:
                """Verify behavior of Mesh."""

                def __init__(self, *args):
                    """Verify behavior of __init__."""

            class NamedSharding:
                """Verify behavior of NamedSharding."""

                def __init__(self, *args):
                    """Verify behavior of __init__."""

            class PartitionSpec:
                """Verify behavior of PartitionSpec."""

                def __init__(self, *args):
                    """Verify behavior of __init__."""

        @staticmethod
        def devices():
            """Verify behavior of devices."""
            return ["cpu"]

        @staticmethod
        def device_put(x, params=None):
            """Verify behavior of device_put."""
            return x

    class MockJnp:
        """Verify behavior of MockJnp."""

        @staticmethod
        def mean(x):
            """Verify behavior of mean."""
            return MockJnpTensor(0.5)

    class MockOptax:
        """Verify behavior of MockOptax."""

        @staticmethod
        def warmup_cosine_decay_schedule(**kwargs):
            """Verify behavior of warmup_cosine_decay_schedule."""
            return "schedule"

        @staticmethod
        def adamw(schedule):
            """Verify behavior of adamw."""
            return "adamw"

        @staticmethod
        def clip_by_global_norm(clip):
            """Verify behavior of clip_by_global_norm."""
            return "clip"

        @staticmethod
        def chain(*args):
            """Verify behavior of chain."""
            return "chain"

        @staticmethod
        def softmax_cross_entropy_with_integer_labels(logits, targets):
            """Verify behavior of softmax_cross_entropy_with_integer_labels."""
            return MockJnpTensor(0.5)

    class MockNNX:
        """Verify behavior of MockNNX."""

        class Rngs:
            """Verify behavior of Rngs."""

            def __init__(self, seed):
                """Verify behavior of __init__."""

        class Optimizer:
            """Verify behavior of Optimizer."""

            def __init__(self, model, tx):
                """Verify behavior of __init__."""

            def update(self, grads):
                """Verify behavior of update."""

        @staticmethod
        def jit(fn):
            """Verify behavior of jit."""
            return fn

        @staticmethod
        def value_and_grad(fn):
            """Verify behavior of value_and_grad."""

            def wrapper(*args, **kwargs):
                """Verify behavior of wrapper."""
                return MockJnpTensor(0.123), "mock_grads"

            return wrapper

    class MockGemma4Config:
        """Verify behavior of MockGemma4Config."""

        @staticmethod
        def gemma4_e2b():
            """Verify behavior of gemma4_e2b."""
            return "mock_config"

    class MockGemma4ForCausalLM:
        """Verify behavior of MockGemma4ForCausalLM."""

        def __init__(self, config, rngs):
            """Verify behavior of __init__."""

        def __call__(self, inputs):
            """Verify behavior of __call__."""
            return MockJnpTensor(0.99)

    monkeypatch.setattr(tr, "jax", MockJax())
    monkeypatch.setattr(tr, "jnp", MockJnp())
    monkeypatch.setattr(tr, "optax", MockOptax())
    monkeypatch.setattr(tr, "nnx", MockNNX())
    monkeypatch.setattr(tr, "Gemma4Config", MockGemma4Config)
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", MockGemma4ForCausalLM)

    def mock_build_dataloader(*args, **kwargs):
        """Verify behavior of mock_build_dataloader."""
        return {"loader": [{"inputs": [1], "targets": [1]}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)


@pytest.mark.usefixtures("_mock_jax_train_env")
def test_train_model_jax_real() -> None:
    """Verify behavior of test_train_model_jax_real."""
    res = tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=1, learning_rate=0.1))
    assert res["backend"] == "jax"
    assert res["status"] == "completed"
    assert res["final_loss"] == pytest.approx(0.123)


@pytest.mark.usefixtures("_mock_jax_train_env")
def test_train_model_jax_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify behavior of test_train_model_jax_error."""

    def mock_raise_error(*args, **kwargs):
        """Verify behavior of mock_raise_error."""
        raise ValueError("mock build dataloader error")

    monkeypatch.setattr(tr, "build_dataloader", mock_raise_error)
    res = tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=1, learning_rate=0.1))
    assert res["status"] == "failed: mock build dataloader error"


def test_train_model_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify behavior of test_train_model_jax_missing."""
    monkeypatch.setattr(tr, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing for training\."):
        tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=1, learning_rate=0.1))


@pytest.mark.usefixtures("_mock_jax_train_env")
def test_execute_train_no_loader_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify behavior of test_execute_train_no_loader_fallback."""

    def mock_build_dataloader(*args, **kwargs):
        """Verify behavior of mock_build_dataloader."""
        return {"loader": None}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    with pytest.raises(ValueError, match="Invalid dataloader"):
        tr._execute_train("dat", 1, 0.1)


@pytest.mark.usefixtures("_mock_jax_train_env")
def test_loss_fn() -> None:
    """Verify behavior of test_loss_fn."""

    def model(x):
        """Verify behavior of model."""
        return MockJnpTensor(0.99)

    batch = {"inputs": [1], "targets": [1]}
    loss = tr._loss_fn(model, batch)
    assert loss.val == pytest.approx(0.5)


@pytest.mark.usefixtures("_mock_jax_train_env")
def test_get_train_step_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify behavior of test_get_train_step_fn."""
    step_fn = tr._get_train_step_fn()

    class MockOpt:
        """Verify behavior of MockOpt."""

        def update(self, grads):
            """Verify behavior of update."""
            self.grads = grads

    opt = MockOpt()
    loss = step_fn(None, opt, {})
    assert loss.val == pytest.approx(0.123)
    assert opt.grads == "mock_grads"

    # Test fallback branches when nnx missing methods
    class MockNNXNoGrad:
        """Verify behavior of MockNNXNoGrad."""

        @staticmethod
        def jit(fn):
            """Verify behavior of jit."""
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
    """Verify behavior of test_run_training_epochs."""

    def mock_train_step(model, opt, batch):
        """Verify behavior of mock_train_step."""
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


def test_real_tiny_jax_model_loss_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _loss_fn computes finite cross-entropy loss on a real tiny JAX Gemma4ForCausalLM."""
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx

    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM

    monkeypatch.setattr(tr, "jax", jax)
    monkeypatch.setattr(tr, "jnp", jnp)
    monkeypatch.setattr(tr, "optax", optax)

    cfg = Gemma4Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
    )
    model = Gemma4ForCausalLM(cfg, rngs=nnx.Rngs(0))
    positions = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    batch = {
        "inputs": jnp.array([[1, 2, 3]], dtype=jnp.int32),
        "targets": jnp.array([[2, 3, 4]], dtype=jnp.int32),
    }
    loss = tr._loss_fn(lambda x: model(x, positions), batch)
    assert float(loss) > 0.0


def test_execute_train_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify behavior of test_execute_train_missing."""
    monkeypatch.setattr(tr, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing for training\."):
        tr._execute_train("dat", 1, 0.1)
    assert tr._resolve_gemma4_config("google/gemma-4-E4B").num_hidden_layers == 42
