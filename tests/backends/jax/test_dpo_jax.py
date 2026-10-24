"""Unit and behavioral tests for JAX Direct Preference Optimization (DPO)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.jax.dpo as jax_dpo
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import DPOConfig, TrainerState
from tests.conftest import requires_jax

pytestmark = requires_jax


def test_real_tiny_jax_dpo_loss_logps_and_train_step() -> None:
    """Verify _compute_logps, dpo_loss, and _get_train_step_fn execute real gradient updates on a tiny Gemma4ForCausalLM."""
    import jax.numpy as jnp
    import optax
    from flax import nnx

    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM

    cfg = Gemma4Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=1,
    )
    policy_model: Any = Gemma4ForCausalLM(cfg, rngs=nnx.Rngs(0))
    ref_model: Any = Gemma4ForCausalLM(cfg, rngs=nnx.Rngs(1))
    optimizer = jax_dpo._create_nnx_optimizer(policy_model, optax.adamw(1e-2))

    batch: dict[str, Any] = {
        "chosen_inputs": jnp.array([[1, 2, 3]], dtype=jnp.int32),
        "chosen_labels": jnp.array([[2, 3, 0]], dtype=jnp.int32),
        "rejected_inputs": jnp.array([[1, 5, 6]], dtype=jnp.int32),
        "rejected_labels": jnp.array([[5, 6, 0]], dtype=jnp.int32),
    }

    old_head = jnp.array(policy_model.lm_head.kernel.value)
    step_loss = jax_dpo._dpo_step_loss(policy_model, ref_model, batch, 0.1)
    assert float(step_loss) > 0.0

    train_step = jax_dpo._get_train_step_fn(0.1)
    state: Any = TrainerState(
        dataloader=[batch],
        epochs=1,
        policy_model=policy_model,
        ref_model=ref_model,
        optimizer=optimizer,
        train_step=train_step,
    )
    final_loss = jax_dpo._run_training_epochs(state)
    assert final_loss > 0.0
    new_head = jnp.array(policy_model.lm_head.kernel.value)
    assert not jnp.allclose(old_head, new_head)


def test_compute_logps_single_arg_model() -> None:
    """Verify _compute_logps falls back to calling model(inputs) when model does not accept positions."""
    import jax.numpy as jnp

    def single_arg_model(inputs: Any) -> Any:
        """Return uniform logits for a 1-arg model."""
        return jnp.ones((*inputs.shape, 8), dtype=jnp.float32)

    inputs = jnp.array([[1, 2]], dtype=jnp.int32)
    labels = jnp.array([[2, 0]], dtype=jnp.int32)
    model_any: Any = single_arg_model
    logps = jax_dpo._compute_logps(model_any, inputs, labels)
    assert logps.shape == (1,)


def test_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dpo_loss and run_dpo handle missing JAX dependencies."""
    monkeypatch.setattr(jax_dpo, "jax", None)
    monkeypatch.setattr(jax_dpo, "jnn", None)
    monkeypatch.setattr(jax_dpo, "jnp", None)

    none_arg: Any = None
    loss, _, _ = jax_dpo.dpo_loss(none_arg, none_arg, none_arg, none_arg)
    assert loss == 0.0

    with pytest.raises(DependencyMissingError, match=r"JAX DPO dependencies are missing\."):
        jax_dpo.run_dpo(DPOConfig(model_name="m", dataset="d"))


def test_get_train_step_fn_without_nnx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_train_step_fn and optimizer helpers handle legacy 1-arg and missing-attribute objects."""
    monkeypatch.setattr(jax_dpo, "nnx", MagicMock(spec=[]))
    with pytest.raises(DependencyMissingError, match="value_and_grad"):
        jax_dpo._get_train_step_fn(0.1)

    # Test _apply_nnx_optimizer_update fallback to 1-arg update(grads)
    called_with: list[Any] = []

    class LegacyOptimizer:
        """Optimizer stub accepting only a single `grads` argument."""

        def update(self, grads: Any) -> None:
            """Record single-argument update call."""
            called_with.append(grads)

    jax_dpo._apply_nnx_optimizer_update(LegacyOptimizer(), "model", "grads")
    assert called_with == ["grads"]

    # Test _create_nnx_optimizer fallback when wrt is rejected and _get_train_step_fn without jit
    class LegacyNNX:
        """NNX module stub whose Optimizer rejects wrt keyword argument and lacks jit."""

        Param = object

        @staticmethod
        def Optimizer(model: Any, tx: Any) -> tuple[Any, Any]:
            """Accept only 2 positional arguments."""
            return (model, tx)

        @staticmethod
        def value_and_grad(fn: Any) -> Any:
            """Invoke fn and return loss and dummy grads."""
            return lambda p, r, b: (fn(p, r, b), "grads")

    monkeypatch.setattr(jax_dpo, "nnx", LegacyNNX)
    assert jax_dpo._create_nnx_optimizer("m", "tx") == ("m", "tx")
    with patch.object(jax_dpo, "_dpo_step_loss", return_value=0.125):
        step_fn: Any = jax_dpo._get_train_step_fn(0.1)
        assert step_fn("p", "r", LegacyOptimizer(), {}) == 0.125


def test_run_training_epochs_float_return() -> None:
    """Verify _run_training_epochs handles plain float returns from train_step."""
    state: Any = TrainerState(dataloader=[{"a": 1}], epochs=1, train_step=lambda *_a: 0.25)
    assert jax_dpo._run_training_epochs(state) == 0.25


def test_execute_dpo_and_run_dpo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _execute_dpo and run_dpo orchestrate model creation, dataloader validation, and error capture."""
    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config

    assert jax_dpo._resolve_gemma4_config("google/gemma-4-E2B") is not None

    tiny_cfg = Gemma4Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=1,
    )
    monkeypatch.setattr(jax_dpo, "_resolve_gemma4_config", lambda _m: tiny_cfg)

    with (
        patch("gemma_4_sql.backends.jax.dpo.build_dataloader", return_value={"loader": ["batch1"]}),
        patch.object(jax_dpo, "_run_training_epochs", return_value=0.5),
    ):
        status, loss = jax_dpo._execute_dpo("model", "dataset", 0.1, 1, 1e-5)
        assert status == "completed"
        assert loss == 0.5

    with (
        patch("gemma_4_sql.backends.jax.dpo.build_dataloader", return_value={"loader": None}),
        pytest.raises(ValueError, match="Invalid dataloader"),
    ):
        jax_dpo._execute_dpo("model", "dataset", 0.1, 1, 1e-5)

    cfg = DPOConfig(model_name="model", dataset="dataset")
    with patch.object(jax_dpo, "_execute_dpo", return_value=("completed", 0.5)):
        res = jax_dpo.run_dpo(cfg)
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5

    with patch.object(jax_dpo, "_execute_dpo", side_effect=ValueError("Test Error")):
        res_err = jax_dpo.run_dpo(cfg)
        assert res_err["status"] == "failed: Test Error"
        with pytest.raises(TrainingError, match="Test Error"):
            jax_dpo.run_dpo(cfg, raise_on_error=True)

    monkeypatch.setattr(jax_dpo, "optax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing for DPO\."):
        jax_dpo._execute_dpo("model", "dataset", 0.1, 1, 1e-5)
