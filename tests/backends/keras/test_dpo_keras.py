"""Unit and behavioral tests for Keras Direct Preference Optimization (DPO)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from gemma_4_sql.backends.keras import dpo
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import DPOConfig, TrainerState
from tests.conftest import requires_keras

pytestmark = requires_keras


def test_real_keras_dpo_loss_logps_and_train_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Keras dpo_loss, _compute_logps, and _get_train_step_fn with a real TensorFlow model and tensors."""
    tf = pytest.importorskip("tensorflow")

    monkeypatch.setattr(dpo.tf, "function", lambda fn, **_kw: fn)

    class TinyTFModel:
        """Minimal TensorFlow causal LM for DPO gradient testing."""

        def __init__(self, seed: int = 0) -> None:
            """Initialize TinyTFModel with trainable embedding and projection weights."""
            tf.random.set_seed(seed)
            self.embed = tf.Variable(tf.random.normal((32, 8), stddev=0.1), trainable=True)
            self.head = tf.Variable(tf.random.normal((8, 32), stddev=0.1), trainable=True)
            self.trainable_variables = [self.embed, self.head]

        def __call__(self, x: Any) -> Any:
            """Compute vocabulary logits from token IDs."""
            h = tf.gather(self.embed, x)
            return tf.matmul(h, self.head)

    policy_model = TinyTFModel(seed=1)
    ref_model = TinyTFModel(seed=2)

    batch = {
        "chosen_inputs": tf.constant([[1, 2, 3]], dtype=tf.int32),
        "chosen_labels": tf.constant([[2, 3, 0]], dtype=tf.int32),
        "rejected_inputs": tf.constant([[1, 4, 5]], dtype=tf.int32),
        "rejected_labels": tf.constant([[4, 5, 0]], dtype=tf.int32),
    }

    pm_any: Any = policy_model
    rm_any: Any = ref_model
    logps = dpo._compute_logps(pm_any, batch["chosen_inputs"], batch["chosen_labels"])
    assert tuple(logps.shape) == (1,)
    assert float(logps.numpy()[0]) < 0.0

    # Also test _compute_logps when model returns an object with .detach().cpu().numpy()
    import numpy as np

    class TorchLikeTensor:
        """Tensor stub mimicking a PyTorch MPS tensor."""

        def detach(self) -> TorchLikeTensor:
            """Return detached tensor stub."""
            return self

        def cpu(self) -> TorchLikeTensor:
            """Return CPU tensor stub."""
            return self

        def numpy(self) -> Any:
            """Return numpy float32 array."""
            return np.zeros((1, 3, 32), dtype=np.float32)

    torch_like_fn: Any = lambda _x: TorchLikeTensor()
    logps_torch_like = dpo._compute_logps(torch_like_fn, batch["chosen_inputs"], batch["chosen_labels"])
    assert tuple(logps_torch_like.shape) == (1,)

    applied_grads: list[Any] = []

    class SimpleTFOptimizer:
        """Minimal optimizer applying TensorFlow gradients."""

        def apply_gradients(self, grads_and_vars: Any) -> None:
            """Apply gradient updates to variables."""
            for g, v in grads_and_vars:
                dense_g = tf.convert_to_tensor(g)
                v.assign_sub(0.05 * dense_g)
                applied_grads.append(dense_g)

    optimizer = SimpleTFOptimizer()
    step_fn: Any = dpo._get_train_step_fn(pm_any, rm_any, optimizer, 0.1)
    loss = step_fn(batch)
    assert float(loss.numpy().mean()) > 0.0
    assert len(applied_grads) == 2

    state: Any = TrainerState(dataloader=[batch], epochs=1, train_step=step_fn)
    final_loss = dpo._run_training_epochs(state)
    assert final_loss > 0.0


def test_dpo_loss_and_logps_missing_tf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dpo_loss, _compute_logps, and _get_train_step_fn fallbacks when TensorFlow is None."""
    monkeypatch.setattr(dpo, "tf", None)
    t_any: Any = 1
    assert dpo.dpo_loss(t_any, t_any, t_any, t_any) == (0.0, 0.0, 0.0)
    assert dpo._compute_logps(MagicMock(), t_any, t_any) == 0.0
    dummy_step: Any = dpo._get_train_step_fn(MagicMock(), MagicMock(), MagicMock(), 0.1)
    assert dummy_step({}) == 0.0


def test_run_training_epochs_loss_variants() -> None:
    """Verify _run_training_epochs handles float and non-numeric step return values."""
    state_float: Any = TrainerState(dataloader=[{"b": 1}], epochs=1, train_step=lambda _b: 2.0)
    assert dpo._run_training_epochs(state_float) == 2.0

    state_str: Any = TrainerState(dataloader=[{"b": 1}], epochs=1, train_step=lambda _b: "non_numeric")
    assert dpo._run_training_epochs(state_str) == 0.0


def test_execute_dpo_and_run_dpo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _execute_dpo and run_dpo handle missing dependencies, preset loading, and dataloader validation."""
    monkeypatch.setattr(dpo, "keras", None)
    with pytest.raises(DependencyMissingError):
        dpo._execute_dpo("m", "d", 0.1, 1, 1e-4)

    mock_keras = MagicMock()
    monkeypatch.setattr(dpo, "keras", mock_keras)
    monkeypatch.setattr(dpo, "tf", MagicMock())

    monkeypatch.setattr(dpo, "GemmaCausalLM", None)
    monkeypatch.setattr(dpo, "optional_attr", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="Failed to load Keras model"):
        dpo._execute_dpo("m", "d", 0.1, 1, 1e-4)

    mock_cls = MagicMock()
    mock_cls.from_preset.return_value = MagicMock()
    monkeypatch.setattr(dpo, "GemmaCausalLM", mock_cls)

    monkeypatch.setattr(dpo, "build_dataloader", MagicMock(return_value={"loader": None}))
    with pytest.raises(ValueError, match="Invalid dataloader"):
        dpo._execute_dpo("m", "d", 0.1, 1, 1e-4)

    monkeypatch.setattr(dpo, "build_dataloader", MagicMock(return_value={"loader": [{"b": 1}]}))
    monkeypatch.setattr(dpo, "_run_training_epochs", MagicMock(return_value=1.5))
    status, loss = dpo._execute_dpo("m", "d", 0.1, 1, 1e-4)
    assert status == "completed"
    assert loss == 1.5

    cfg = DPOConfig(model_name="m", dataset="d")
    monkeypatch.setattr(dpo, "keras", None)
    with pytest.raises(DependencyMissingError):
        dpo.run_dpo(cfg)

    monkeypatch.setattr(dpo, "keras", MagicMock())
    monkeypatch.setattr(dpo, "tf", MagicMock())
    monkeypatch.setattr(dpo, "_execute_dpo", MagicMock(return_value=("completed", 0.75)))
    res_ok = dpo.run_dpo(cfg)
    assert res_ok["status"] == "completed"
    assert res_ok["final_loss"] == 0.75

    monkeypatch.setattr(dpo, "_execute_dpo", MagicMock(side_effect=ValueError("simulated dpo failure")))
    res_err = dpo.run_dpo(cfg)
    assert "failed: simulated dpo failure" in str(res_err["status"])
    with pytest.raises(TrainingError, match="simulated dpo failure"):
        dpo.run_dpo(cfg, raise_on_error=True)
