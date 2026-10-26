"""Unit and behavioral tests for common DPO utilities."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from gemma_4_sql.backends.common_dpo import generic_dpo_loss, generic_run_training_epochs, run_dpo_wrapper
from gemma_4_sql.exceptions import TrainingError
from gemma_4_sql.type_hints import DPOConfig, DPOLoopState


class DummyTensor:
    """Minimal numeric tensor stub supporting detach, mean, item, and arithmetic."""

    def __init__(self, value: float) -> None:
        """Initialize tensor stub with a float value."""
        self.value = value

    def __sub__(self, other: Any) -> DummyTensor:
        """Subtract another DummyTensor."""
        return DummyTensor(self.value - float(other.value))

    def __mul__(self, other: Any) -> DummyTensor:
        """Multiply by scalar float."""
        return DummyTensor(self.value * float(other))

    def __rmul__(self, other: Any) -> DummyTensor:
        """Right-multiply by scalar float."""
        return self.__mul__(other)

    def __neg__(self) -> DummyTensor:
        """Negate value."""
        return DummyTensor(-self.value)

    def detach(self) -> DummyTensor:
        """Return detached copy."""
        return DummyTensor(self.value)

    def mean(self) -> float:
        """Return scalar value as mean."""
        return self.value

    def item(self) -> float:
        """Return scalar float value."""
        return self.value


def test_generic_dpo_loss_with_methods() -> None:
    """Verify generic_dpo_loss computes DPO loss and detached rewards when detach/mean exist."""
    policy_chosen: Any = DummyTensor(1.0)
    policy_rejected: Any = DummyTensor(0.5)
    ref_chosen: Any = DummyTensor(0.8)
    ref_rejected: Any = DummyTensor(0.4)
    beta = 0.1

    def log_sigmoid(x: Any) -> DummyTensor:
        """Scale input value by 2."""
        return DummyTensor(float(x.value) * 2.0)

    loss, chosen_rewards, rejected_rewards = generic_dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta,
        log_sigmoid,
    )
    cr_any: Any = chosen_rewards
    rr_any: Any = rejected_rewards

    assert loss == pytest.approx(-0.02)
    assert cr_any.value == pytest.approx(0.1 * (1.0 - 0.8))
    assert rr_any.value == pytest.approx(0.1 * (0.5 - 0.4))


def test_generic_dpo_loss_without_methods() -> None:
    """Verify generic_dpo_loss works on plain scalar numbers lacking detach/mean methods."""
    policy_chosen: Any = 1.0
    policy_rejected: Any = 0.5
    ref_chosen: Any = 0.8
    ref_rejected: Any = 0.4
    beta = 0.1

    def log_sigmoid(x: Any) -> float:
        """Scale float by 2."""
        return float(x) * 2.0

    loss, chosen_rewards, rejected_rewards = generic_dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta,
        log_sigmoid,
    )

    assert loss == pytest.approx(-0.02)
    assert chosen_rewards == pytest.approx(0.02)
    assert rejected_rewards == pytest.approx(0.01)


def test_generic_run_training_epochs_none_dataloader() -> None:
    """Verify generic_run_training_epochs returns 0.0 when dataloader is None."""
    state: DPOLoopState[Any, Any, Any] = DPOLoopState(dataloader=None, epochs=2)
    assert generic_run_training_epochs(state, lambda *_a: 0.0) == 0.0


def test_generic_run_training_epochs_normal() -> None:
    """Verify generic_run_training_epochs averages batch losses with .item()."""
    state: DPOLoopState[Any, Any, Any] = DPOLoopState(
        dataloader=[{"data": 1}, {"data": 2}],
        epochs=2,
        beta=0.1,
    )

    def step_fn(_policy: Any, _ref: Any, _opt: Any, batch: Any, _beta: float) -> DummyTensor:
        """Return DummyTensor proportional to batch data."""
        return DummyTensor(float(batch["data"]) * 0.5)

    final_loss = generic_run_training_epochs(state, step_fn)
    assert final_loss == 0.75


def test_generic_run_training_epochs_no_item() -> None:
    """Verify generic_run_training_epochs handles plain float batch losses and generator dataloaders."""

    class GeneratorLoader:
        """Iterable dataloader without __len__."""

        def __iter__(self) -> Iterator[dict[str, int]]:
            """Yield two batches."""
            yield {"data": 1}
            yield {"data": 2}

    state: DPOLoopState[Any, Any, Any] = DPOLoopState(dataloader=GeneratorLoader(), epochs=1, beta=0.1)

    def step_fn(_policy: Any, _ref: Any, _opt: Any, batch: Any, _beta: float) -> float:
        """Return plain float loss."""
        return float(batch["data"]) * 0.5

    final_loss = generic_run_training_epochs(state, step_fn)
    assert final_loss == 0.75


def test_run_dpo_wrapper() -> None:
    """Verify run_dpo_wrapper executes DPO callable, captures errors, and raises TrainingError when requested."""
    cfg = DPOConfig(model_name="m", dataset="d", beta=0.2, epochs=2, learning_rate=1e-4, batch_size=4)
    res_ok = run_dpo_wrapper("jax", cfg, lambda _m, _d, _b, _ep, _lr, _bs: ("completed", 0.125))
    assert res_ok["status"] == "completed"
    assert res_ok["final_loss"] == 0.125

    def _fail(*_a: Any) -> tuple[str, float]:
        """Raise ValueError for testing error handling."""
        raise ValueError("dpo boom")

    res_fail = run_dpo_wrapper("jax", cfg, _fail)
    assert res_fail["status"] == "failed: dpo boom"

    with pytest.raises(TrainingError, match="dpo boom"):
        run_dpo_wrapper("jax", cfg, _fail, raise_on_error=True)

    from gemma_4_sql.exceptions import DependencyMissingError

    with pytest.raises(DependencyMissingError, match="missing dpo dep"):
        run_dpo_wrapper("jax", cfg, lambda *_a: (_ for _ in ()).throw(DependencyMissingError("missing dpo dep")))
