"""Unit and behavioral tests for the JAX training pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.jax.train as tr
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import TrainerState, TrainingConfig
from tests.conftest import requires_jax

pytestmark = requires_jax


def test_real_tiny_jax_training_step_and_train_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _loss_fn, _get_train_step_fn, and train_model execute real gradient updates on a tiny Gemma4ForCausalLM."""
    import jax.numpy as jnp

    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config

    tiny_cfg = Gemma4Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=1,
    )
    monkeypatch.setattr(tr, "_resolve_gemma4_config", lambda _m: tiny_cfg)

    batch = {
        "inputs": jnp.array([[1, 2, 3]], dtype=jnp.int32),
        "targets": jnp.array([[2, 3, 4]], dtype=jnp.int32),
    }
    monkeypatch.setattr(tr, "build_dataloader", lambda *_a, **_k: {"loader": [batch]})

    res = tr.train_model(TrainingConfig(action="sft", model_name="tiny", dataset="dat", epochs=1, learning_rate=1e-2))
    assert res["backend"] == "jax"
    assert res["status"] == "completed"
    assert res["final_loss"] > 0.0


def test_loss_fn_single_arg_model() -> None:
    """Verify _loss_fn falls back to 1-arg model(inputs) when positions argument is rejected."""
    import jax.numpy as jnp

    def single_arg_model(inputs: Any) -> Any:
        """Return uniform logits for a 1-arg model."""
        return jnp.ones((*inputs.shape, 8), dtype=jnp.float32)

    batch = {
        "inputs": jnp.array([[1, 2]], dtype=jnp.int32),
        "targets": jnp.array([[2, 3]], dtype=jnp.int32),
    }
    model_any: Any = single_arg_model
    loss = tr._loss_fn(model_any, batch)
    assert float(loss) > 0.0


def test_train_model_jax_error_and_invalid_dataloader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify train_model and _execute_train handle invalid dataloaders and runtime exceptions."""
    from gemma_4_sql.backends.jax.gemma4 import Gemma4Config

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
    monkeypatch.setattr(tr, "_resolve_gemma4_config", lambda _m: tiny_cfg)
    monkeypatch.setattr(tr, "build_dataloader", lambda *_a, **_k: {"loader": None})

    with pytest.raises(ValueError, match="Invalid dataloader"):
        tr._execute_train("dat", 1, 0.1)

    res = tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=1, learning_rate=0.1))
    assert "failed: Invalid dataloader" in res["status"]

    with pytest.raises(TrainingError, match="Invalid dataloader") as exc_info:
        tr.train_model(
            TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=1, learning_rate=0.1),
            raise_on_error=True,
        )
    assert exc_info.value.backend == "jax"
    assert exc_info.value.model_name == "mod"


def test_get_train_step_fn_and_optimizer_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_train_step_fn, _create_nnx_optimizer, and _apply_nnx_optimizer_update fallback branches."""
    called_with: list[Any] = []

    class LegacyOptimizer:
        """Optimizer stub accepting only a single `grads` argument."""

        def update(self, grads: Any) -> None:
            """Record single-argument update call."""
            called_with.append(grads)

    tr._apply_nnx_optimizer_update(LegacyOptimizer(), "model", "grads")
    assert called_with == ["grads"]

    class LegacyNNX:
        """NNX module stub whose Optimizer rejects wrt keyword argument and lacks jit."""

        Param = object

        @staticmethod
        def Optimizer(model: Any, tx: Any) -> tuple[Any, Any]:
            """Accept only 2 positional arguments."""
            return (model, tx)

        @staticmethod
        def value_and_grad(fn: Any) -> Any:
            """Return a dummy value_and_grad callable."""
            return lambda *_a: (0.25, "grads")

    monkeypatch.setattr(tr, "nnx", LegacyNNX)
    assert tr._create_nnx_optimizer("m", "tx") == ("m", "tx")
    step_fn_no_jit: Any = tr._get_train_step_fn()
    assert step_fn_no_jit("m", LegacyOptimizer(), {}) == 0.25

    monkeypatch.setattr(tr, "nnx", MagicMock(spec=[]))
    with pytest.raises(DependencyMissingError, match="value_and_grad"):
        tr._get_train_step_fn()

    monkeypatch.setattr(tr, "nnx", None)
    with pytest.raises(DependencyMissingError, match="value_and_grad"):
        tr._get_train_step_fn()


def test_run_training_epochs_float_return() -> None:
    """Verify _run_training_epochs handles plain float returns from train_step."""
    state: Any = TrainerState(
        dataloader=[{"inputs": [1], "targets": [1]}],
        epochs=1,
        policy_model=None,
        optimizer=None,
        train_step=lambda *_a: 0.42,
        params=None,
    )
    with patch.object(tr.jax, "device_put", side_effect=lambda x, _p: x):
        assert tr._run_training_epochs(state) == pytest.approx(0.42)


def test_execute_train_and_train_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify train_model and _execute_train raise DependencyMissingError when JAX is missing."""
    assert tr._resolve_gemma4_config("google/gemma-4-E4B").num_hidden_layers == 42
    monkeypatch.setattr(tr, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing for training\."):
        tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=1, learning_rate=0.1))
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing for training\."):
        tr._execute_train("dat", 1, 0.1)
