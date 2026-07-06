"""Test JAX DPO logic."""

import pytest

import gemma_4_sql.backends.jax.dpo as jax_dpo
from gemma_4_sql.type_hints import DPOConfig, TrainerState


class MockJnp:
    """Provide class docstring."""

    int32 = "int32"

    def zeros(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""

        class MockZero:
            """Class docstring."""

            def __sub__(self, other):
                """Test function."""
                return self

            def __rsub__(self, other):
                """Test function."""
                return self

            def __mul__(self, other):
                """Test function."""
                return self

        return MockZero()

    def expand_dims(self, a: object, axis: int) -> object:
        """Execute function."""
        return a

    def take_along_axis(self, a: object, indices: object, axis: int) -> object:
        """Execute function."""
        return a

    def squeeze(self, a: object, axis: int) -> object:
        """Execute function."""
        return a

    def sum(self, a: object, axis: int) -> object:
        """Execute function."""
        return a


class MockJnn:
    """Provide class docstring."""

    def log_softmax(self, logits: object, axis: int) -> object:
        """Execute function."""
        return logits

    def log_sigmoid(self, x: object) -> object:
        """Execute function."""
        return x


class MockModel:
    """Class docstring."""

    def __call__(self, inputs: object) -> object:
        """Test function."""
        return inputs


def test_compute_logps_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test compute logps in jax."""
    monkeypatch.setattr(jax_dpo, "jnp", MockJnp())
    monkeypatch.setattr(jax_dpo, "jnn", MockJnn())

    inputs = [1, 2]
    labels = [1, 2]
    model = MockModel()
    res = jax_dpo._compute_logps(model, inputs, labels)
    assert res is not None


def test_dpo_loss_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function."""
    monkeypatch.setattr(jax_dpo, "jnp", None)
    monkeypatch.setattr(jax_dpo, "jnn", None)
    res = jax_dpo.dpo_loss(0, 0, 0, 0)
    assert res == (0.0, 0.0, 0.0)

    monkeypatch.setattr(jax_dpo, "jnp", MockJnp())
    monkeypatch.setattr(jax_dpo, "jnn", MockJnn())
    res = jax_dpo.dpo_loss(0, 0, 0, 0)
    assert res is not None


def test_dpo_step_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test dpo step loss."""
    monkeypatch.setattr(jax_dpo, "jnp", MockJnp())
    monkeypatch.setattr(jax_dpo, "jnn", MockJnn())

    batch = {"chosen_inputs": 1, "chosen_labels": 1, "rejected_inputs": 1, "rejected_labels": 1}
    res = jax_dpo._dpo_step_loss(MockModel(), MockModel(), batch, 0.1)
    assert res is not None


def test_get_train_step_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get train step fn."""

    class MockNnx:
        """Class docstring."""

        def jit(self, fn: object) -> object:
            """Test function."""
            return fn

        def value_and_grad(self, fn: object) -> object:
            """Test function."""

            class MockLossReturn:
                """Class docstring."""

                def item(self):
                    """Test function."""
                    return 1.0

            return lambda *args: (MockLossReturn(), 1)

    monkeypatch.setattr(jax_dpo, "nnx", MockNnx())
    monkeypatch.setattr(jax_dpo, "jnp", MockJnp())
    monkeypatch.setattr(jax_dpo, "jnn", MockJnn())

    class MockOpt:
        """Class docstring."""

        def update(self, grads: object) -> None:
            """Test function."""

    fn = jax_dpo._get_train_step_fn(0.1)
    batch = {"chosen_inputs": 1, "chosen_labels": 1, "rejected_inputs": 1, "rejected_labels": 1}
    res = fn(MockModel(), MockModel(), MockOpt(), batch)
    assert res is not None


def test_run_training_epochs() -> None:
    """Test run training epochs."""

    class MockLoss:
        """Class docstring."""

        def item(self):
            """Test function."""
            return 1.0

    state = TrainerState(dataloader=[1, 2], epochs=2, policy_model=None, ref_model=None, optimizer=None, train_step=lambda *a: MockLoss())
    res = jax_dpo._run_training_epochs(state)
    assert res == 1.0


def test_run_dpo_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run dpo mocked."""
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(jax_dpo, "jax", None)
    with pytest.raises(DependencyMissingError, match="JAX DPO dependencies are missing."):
        jax_dpo.run_dpo(DPOConfig(model_name="m", dataset="d"))


def test_run_dpo_real_no_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run dpo real no loader."""
    monkeypatch.setattr(jax_dpo, "jax", object())
    monkeypatch.setattr(jax_dpo, "jnp", MockJnp())
    monkeypatch.setattr(jax_dpo, "jnn", MockJnn())
    monkeypatch.setattr(jax_dpo, "optax", type("MockOptax", (), {"adamw": lambda x: x}))

    monkeypatch.setattr(jax_dpo, "Gemma4ForCausalLM", lambda *a, **k: MockModel())
    monkeypatch.setattr(jax_dpo, "Gemma4Config", type("MockConfig", (), {"gemma4_e2b": lambda: None}))

    class MockNnx:
        """Class docstring."""

        def jit(self, fn: object) -> object:
            """Test function."""
            return fn

        def value_and_grad(self, fn: object) -> object:
            """Test function."""

            class MockLossReturn:
                """Class docstring."""

                def item(self):
                    """Test function."""
                    return 1.0

            return lambda *args: (MockLossReturn(), 1)

        class Rngs:
            """Class docstring."""

            def __init__(self, _):
                """Test function."""

        class Optimizer:
            """Class docstring."""

            def __init__(self, *a):
                """Test function."""

            def update(self, grads: object) -> None:
                """Test function."""

    monkeypatch.setattr(jax_dpo, "nnx", MockNnx())
    monkeypatch.setattr(jax_dpo, "build_dataloader", lambda *a, **k: {})

    res = jax_dpo.run_dpo(DPOConfig(model_name="m", dataset="d"))
    if "failed" in res["status"]:
        raise AssertionError(res["status"])


def test_run_dpo_real_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run dpo real loader."""
    monkeypatch.setattr(jax_dpo, "jax", object())
    monkeypatch.setattr(jax_dpo, "jnp", MockJnp())
    monkeypatch.setattr(jax_dpo, "jnn", MockJnn())
    monkeypatch.setattr(jax_dpo, "optax", type("MockOptax", (), {"adamw": lambda x: x}))

    monkeypatch.setattr(jax_dpo, "Gemma4ForCausalLM", lambda *a, **k: MockModel())
    monkeypatch.setattr(jax_dpo, "Gemma4Config", type("MockConfig", (), {"gemma4_e2b": lambda: None}))

    class MockNnx:
        """Class docstring."""

        def jit(self, fn: object) -> object:
            """Test function."""
            return fn

        def value_and_grad(self, fn: object) -> object:
            """Test function."""

            class MockLossReturn:
                """Class docstring."""

                def item(self):
                    """Test function."""
                    return 1.0

            return lambda *args: (MockLossReturn(), 1)

        class Rngs:
            """Class docstring."""

            def __init__(self, _):
                """Test function."""

        class Optimizer:
            """Class docstring."""

            def __init__(self, *a):
                """Test function."""

            def update(self, grads: object) -> None:
                """Test function."""

    monkeypatch.setattr(jax_dpo, "nnx", MockNnx())
    monkeypatch.setattr(jax_dpo, "build_dataloader", lambda *a, **k: {"loader": [{"chosen_inputs": 1, "chosen_labels": 1, "rejected_inputs": 1, "rejected_labels": 1}]})

    monkeypatch.setattr(jax_dpo, "Gemma4ForCausalLM", lambda *a, **k: MockModel())

    res = jax_dpo.run_dpo(DPOConfig(model_name="m", dataset="d"))
    if res["status"] != "completed":
        raise AssertionError(res["status"])
    assert res["final_loss"] == 1.0


def test_run_dpo_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run dpo error."""
    monkeypatch.setattr(jax_dpo, "jax", object())
    monkeypatch.setattr(jax_dpo, "jnp", MockJnp())
    monkeypatch.setattr(jax_dpo, "jnn", MockJnn())
    monkeypatch.setattr(jax_dpo, "optax", object())
    monkeypatch.setattr(jax_dpo, "Gemma4ForCausalLM", object())

    def raise_err(*a: object, **k: object) -> None:
        """Test function."""
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(jax_dpo, "_execute_dpo", raise_err)
    res = jax_dpo.run_dpo(DPOConfig(model_name="m", dataset="d"))
    if "failed" not in res["status"]:
        raise AssertionError
