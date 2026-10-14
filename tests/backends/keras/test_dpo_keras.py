"""Unit tests for Keras DPO loss calculation, log-probability extraction, and training loop."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from gemma_4_sql.backends.keras import dpo
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import DPOConfig, TrainerState
from tests.conftest import requires_keras

pytestmark = requires_keras


def test_run_dpo_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run_dpo and _execute_dpo raise DependencyMissingError when Keras or TensorFlow are missing."""
    cfg = DPOConfig(model_name="m", dataset="d")

    monkeypatch.setattr(dpo, "keras", None)
    with pytest.raises(DependencyMissingError, match="Keras DPO dependencies are missing"):
        dpo.run_dpo(cfg)
    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing"):
        dpo._execute_dpo("m", "d", 0.1, 1, 1e-4)

    monkeypatch.setattr(dpo, "keras", MagicMock())
    monkeypatch.setattr(dpo, "tf", None)
    with pytest.raises(DependencyMissingError, match="Keras DPO dependencies are missing"):
        dpo.run_dpo(cfg)


def test_dpo_loss_and_compute_logps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dpo_loss and _compute_logps compute expected log-probability reductions."""
    monkeypatch.setattr(dpo, "tf", None)
    assert dpo.dpo_loss(None, None, None, None) == (0.0, 0.0, 0.0)  # type: ignore[arg-type]
    assert dpo._compute_logps(None, None, None) == 0.0  # type: ignore[arg-type]

    mock_tf = MagicMock()

    class FakeTensor:
        """Tensor stub supporting equality comparison."""

        def __init__(self, val: float) -> None:
            """Initialize FakeTensor with float value."""
            self.val = val

        def __eq__(self, other: object) -> bool:
            """Compare underlying float value."""
            return self.val == other

    mock_tf.reduce_sum = lambda x, axis: FakeTensor(-1.0) if x == "case1" else FakeTensor(-1.5)
    mock_tf.cast = lambda x, _y: x
    mock_tf.expand_dims = lambda x, axis: x
    mock_tf.gather = lambda x, _y, batch_dims: x
    mock_tf.squeeze = lambda x, axis: x
    mock_tf.nn.log_softmax = lambda _x, axis: "case1"
    monkeypatch.setattr(dpo, "tf", mock_tf)

    model = MagicMock(return_value=1.0)
    assert dpo._compute_logps(model, MagicMock(), MagicMock()) == -1.0

    mock_tf.nn.log_softmax = lambda _x, axis: "case2"
    assert dpo._compute_logps(model, MagicMock(), MagicMock()) == -1.5

    monkeypatch.setattr("gemma_4_sql.backends.keras.dpo.generic_dpo_loss", MagicMock(return_value=(0.5, 0.1, -0.1)))
    assert dpo.dpo_loss(MagicMock(), MagicMock(), MagicMock(), MagicMock()) == (0.5, 0.1, -0.1)


def test_get_train_step_fn_and_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_train_step_fn builds a step function and _run_training_epochs averages losses."""
    monkeypatch.setattr(dpo, "tf", None)
    step_none = dpo._get_train_step_fn(MagicMock(), MagicMock(), MagicMock(), 0.1)
    assert step_none({}) == 0.0

    mock_tf = MagicMock()

    class MockTape:
        """No-op GradientTape context manager."""

        def __enter__(self) -> MagicMock:
            """Enter GradientTape."""
            return MagicMock()

        def __exit__(self, *_args: object) -> None:
            """Exit GradientTape."""

    mock_tf.GradientTape = MockTape
    mock_tf.function = lambda fn: fn
    monkeypatch.setattr(dpo, "tf", mock_tf)
    monkeypatch.setattr(dpo, "_compute_logps", MagicMock(return_value=1.0))
    monkeypatch.setattr(dpo, "dpo_loss", MagicMock(return_value=(1.25, 0.0, 0.0)))

    step_fn = dpo._get_train_step_fn(MagicMock(), MagicMock(), MagicMock(), 0.1)
    batch: dict[str, Any] = {"chosen_inputs": 1, "chosen_labels": 1, "rejected_inputs": 1, "rejected_labels": 1}
    assert step_fn(batch) == 1.25

    state_np = TrainerState(dataloader=[batch, batch], epochs=1, train_step=lambda _b: MagicMock(numpy=lambda: 1.5))
    assert dpo._run_training_epochs(state_np) == 1.5

    state_float = TrainerState(dataloader=[batch, batch], epochs=1, train_step=lambda _b: 2.0)
    assert dpo._run_training_epochs(state_float) == 2.0

    state_invalid = TrainerState(dataloader=[batch], epochs=1, train_step=lambda _b: "not_a_float")
    assert dpo._run_training_epochs(state_invalid) == 0.0


def test_execute_dpo_and_run_dpo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _execute_dpo and run_dpo handle preset loading, dataloader validation, and execution."""
    monkeypatch.setattr(dpo, "keras", MagicMock())
    monkeypatch.setattr(dpo, "tf", MagicMock())

    monkeypatch.setattr(dpo, "GemmaCausalLM", None)
    monkeypatch.setattr(dpo, "optional_attr", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="Failed to load Keras model"):
        dpo._execute_dpo("m", "d", 0.1, 1, 1e-4)

    mock_gemma_cls = MagicMock()
    monkeypatch.setattr(dpo, "GemmaCausalLM", mock_gemma_cls)
    monkeypatch.setattr(dpo, "_get_train_step_fn", MagicMock())

    monkeypatch.setattr(dpo, "build_dataloader", MagicMock(return_value={}))
    with pytest.raises(ValueError, match="Invalid dataloader"):
        dpo._execute_dpo("m", "d", 0.1, 1, 1e-4)

    monkeypatch.setattr(dpo, "build_dataloader", MagicMock(return_value={"loader": [1, 2]}))
    monkeypatch.setattr(dpo, "_run_training_epochs", MagicMock(return_value=0.75))
    status, loss = dpo._execute_dpo("m", "d", 0.1, 1, 1e-4)
    assert status == "completed"
    assert loss == 0.75

    cfg = DPOConfig(model_name="m", dataset="d", beta=0.1, epochs=1, learning_rate=1e-4)
    res_ok = dpo.run_dpo(cfg)
    assert res_ok["status"] == "completed"
    assert res_ok["final_loss"] == 0.75

    monkeypatch.setattr(dpo, "_execute_dpo", MagicMock(side_effect=ValueError("simulated dpo failure")))
    res_err = dpo.run_dpo(cfg)
    assert "failed: simulated dpo failure" in str(res_err["status"])
