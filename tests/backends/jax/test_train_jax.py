"""Unit and behavioral tests for the JAX training pipeline."""

from __future__ import annotations

from unittest.mock import patch

import jax
import pytest

import gemma_4_sql.backends.jax.train as tr
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import JaxTrainingState, TrainingConfig
from tests.conftest import requires_jax

pytestmark = requires_jax


class _EmptyNNXStub:
    """Stub lacking `value_and_grad` and `jit` to verify `DependencyMissingError`."""


def test_real_tiny_jax_training_step_and_train_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _loss_fn, _get_train_step_fn, and train_model run real masked updates on a tiny Gemma4ForCausalLM."""
    from gemma_4_sql.backends.jax.gemma4 import Gemma4ForCausalLM
    from gemma_4_sql.backends.jax.gemma4._jnp import jnp
    from gemma_4_sql.type_hints import ETLConfig

    batch = {
        "inputs": jnp.array([[2, 5, 6, 7]], dtype=jnp.int32),
        "targets": jnp.array([[5, 6, 7, 1]], dtype=jnp.int32),
        "loss_mask": jnp.array([[0, 0, 1, 1]], dtype=jnp.int32),
    }

    def _batch_loader(*_a: ETLConfig) -> dict[str, list[dict[str, jax.Array]]]:
        """Return 1-batch loader."""
        return {"loader": [batch]}

    monkeypatch.setattr(tr, "build_dataloader", _batch_loader)

    res = tr.train_model(
        TrainingConfig(action="sft", model_name="gemma-4-tiny", dataset="dat", epochs=1, learning_rate=1e-2)
    )
    assert res["backend"] == "jax"
    assert res["status"] == "completed"
    assert res["final_loss"] > 0.0

    # Second run exercises cached model branch in _execute_train
    res2 = tr.train_model(
        TrainingConfig(action="sft", model_name="gemma-4-tiny", dataset="dat", epochs=1, learning_rate=1e-2)
    )
    assert res2["status"] == "completed"
    assert isinstance(tr._MODEL_CACHE.get_model("gemma-4-tiny"), Gemma4ForCausalLM)


def test_loss_fn_masked_and_unshifted_inputs() -> None:
    """Verify _loss_fn masks prompt tokens (`loss_mask == 0`) and shifts `input_ids` when targets are omitted."""
    from gemma_4_sql.backends.jax.gemma4._jnp import jnp

    def single_arg_model(inputs: jax.Array) -> jax.Array:
        """Return deterministic logits for a 1-arg model."""
        vocab = 8
        seq_len = inputs.shape[1]
        base = jnp.zeros((1, seq_len, vocab), dtype=jnp.float32)
        return base.at[0, :, 3].set(10.0)

    # Prompt token has wrong target 7 (high loss if unmasked), SQL token has target 3 (near-zero loss)
    masked_batch = {
        "inputs": jnp.array([[1, 2]], dtype=jnp.int32),
        "targets": jnp.array([[7, 3]], dtype=jnp.int32),
        "loss_mask": jnp.array([[0, 1]], dtype=jnp.int32),
    }
    loss_masked = float(tr._loss_fn(single_arg_model, masked_batch))
    unmasked_batch = {
        "inputs": jnp.array([[1, 2]], dtype=jnp.int32),
        "targets": jnp.array([[7, 3]], dtype=jnp.int32),
    }
    loss_unmasked = float(tr._loss_fn(single_arg_model, unmasked_batch))
    assert loss_masked < 0.01
    assert loss_unmasked > 4.0

    # Unshifted input_ids-only batch shifts logits[:, :-1, :] against input_ids[:, 1:]
    input_only_batch = {
        "input_ids": jnp.array([[2, 3, 3]], dtype=jnp.int32),
        "loss_mask": jnp.array([[0, 1, 1]], dtype=jnp.int32),
    }
    loss_shifted = float(tr._loss_fn(single_arg_model, input_only_batch))
    assert loss_shifted < 0.01


def test_train_model_jax_error_and_invalid_dataloader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify train_model and _execute_train handle invalid dataloaders and runtime exceptions."""
    from gemma_4_sql.type_hints import ETLConfig

    def _none_loader(*_a: ETLConfig) -> dict[str, None]:
        """Return loader=None."""
        return {"loader": None}

    monkeypatch.setattr(tr, "build_dataloader", _none_loader)
    tr._MODEL_CACHE.pop("gemma-4-tiny", None)

    with pytest.raises(ValueError, match="Invalid dataloader"):
        tr._execute_train("dat", 1, 0.1, model_name="gemma-4-tiny")

    res = tr.train_model(
        TrainingConfig(action="sft", model_name="gemma-4-tiny", dataset="dat", epochs=1, learning_rate=0.1)
    )
    assert "failed: Invalid dataloader" in res["status"]

    with pytest.raises(TrainingError, match="Invalid dataloader") as exc_info:
        tr.train_model(
            TrainingConfig(action="sft", model_name="gemma-4-tiny", dataset="dat", epochs=1, learning_rate=0.1),
            raise_on_error=True,
        )
    assert exc_info.value.backend == "jax"
    assert exc_info.value.model_name == "gemma-4-tiny"


def test_get_train_step_fn_and_optimizer_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_train_step_fn raises DependencyMissingError when nnx is missing or lacks value_and_grad."""
    monkeypatch.setattr(tr, "nnx", _EmptyNNXStub())
    with pytest.raises(DependencyMissingError, match="value_and_grad"):
        tr._get_train_step_fn()

    monkeypatch.setattr(tr, "nnx", None)
    with pytest.raises(DependencyMissingError, match="value_and_grad"):
        tr._get_train_step_fn()


def test_run_training_epochs_float_return() -> None:
    """Verify _run_training_epochs handles plain float and non-numeric returns from train_step."""

    def _ident_put(x: list[int], _p: None) -> list[int]:
        """Return `x` unchanged."""
        return x

    def _float_step(*_a: dict[str, list[int]] | None) -> float:
        """Return 0.42."""
        return 0.42

    def _str_step(*_a: dict[str, list[int]] | None) -> str:
        """Return 'non_numeric'."""
        return "non_numeric"

    state: JaxTrainingState[None, None, dict[str, list[int]]] = JaxTrainingState(
        dataloader=[{"inputs": [1], "targets": [1]}],
        epochs=1,
        model=None,
        optimizer=None,
        train_step=_float_step,
        sharding=None,
    )
    with patch.object(tr.jax, "device_put", side_effect=_ident_put):
        assert tr._run_training_epochs(state) == pytest.approx(0.42)
    state_non_num: JaxTrainingState[None, None, dict[str, list[int]]] = JaxTrainingState(
        dataloader=[{"inputs": [1], "targets": [1]}],
        epochs=1,
        model=None,
        optimizer=None,
        train_step=_str_step,
        sharding=None,
    )
    with patch.object(tr.jax, "device_put", side_effect=_ident_put):
        assert tr._run_training_epochs(state_non_num) == 0.0


def test_execute_train_and_train_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify train_model and _execute_train raise DependencyMissingError when JAX is missing."""
    assert tr._resolve_gemma4_config("google/gemma-4-E4B").num_hidden_layers == 42
    monkeypatch.setattr(tr, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing for training\."):
        tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=1, learning_rate=0.1))
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing for training\."):
        tr._execute_train("dat", 1, 0.1)
