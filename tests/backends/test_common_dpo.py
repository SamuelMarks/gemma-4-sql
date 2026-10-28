"""Unit and behavioral tests for common DPO utilities."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from gemma_4_sql.backends.common_dpo import generic_dpo_loss, generic_run_training_epochs, run_dpo_wrapper
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import DPOConfig, DPOLoopState


class DummyTensor:
    """Minimal numeric tensor stub supporting detach, mean, item, float conversion, and arithmetic."""

    def __init__(self, value: float) -> None:
        """Initialize tensor stub with a float value."""
        self.value = value

    def __sub__(self, other: DummyTensor | float, /) -> DummyTensor:
        """Subtract another DummyTensor or float."""
        other_val = other.value if isinstance(other, DummyTensor) else float(other)
        return DummyTensor(self.value - other_val)

    def __rsub__(self, other: DummyTensor | float, /) -> DummyTensor:
        """Right-subtract this DummyTensor from a float or DummyTensor."""
        other_val = other.value if isinstance(other, DummyTensor) else float(other)
        return DummyTensor(other_val - self.value)

    def __mul__(self, other: DummyTensor | float, /) -> DummyTensor:
        """Multiply by scalar float or DummyTensor."""
        other_val = other.value if isinstance(other, DummyTensor) else float(other)
        return DummyTensor(self.value * other_val)

    def __rmul__(self, other: DummyTensor | float, /) -> DummyTensor:
        """Right-multiply by scalar float or DummyTensor."""
        return self.__mul__(other)

    def __neg__(self) -> DummyTensor:
        """Negate value."""
        return DummyTensor(-self.value)

    def detach(self) -> DummyTensor:
        """Return detached copy."""
        return DummyTensor(self.value)

    def mean(self) -> DummyTensor:
        """Return scalar value wrapped in DummyTensor."""
        return DummyTensor(self.value)

    def item(self) -> float:
        """Return scalar float value."""
        return self.value

    def __float__(self) -> float:
        """Convert to Python float."""
        return self.value


def test_generic_dpo_loss_with_methods() -> None:
    """Verify generic_dpo_loss computes DPO loss and detached rewards when detach/mean exist."""
    policy_chosen = DummyTensor(1.0)
    policy_rejected = DummyTensor(0.5)
    ref_chosen = DummyTensor(0.8)
    ref_rejected = DummyTensor(0.4)
    beta = 0.1

    def log_sigmoid(x: DummyTensor) -> DummyTensor:
        """Scale input value by 2."""
        return DummyTensor(x.value * 2.0)

    loss, chosen_rewards, rejected_rewards = generic_dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta,
        log_sigmoid,
    )

    assert float(loss) == pytest.approx(-0.02)
    assert chosen_rewards.value == pytest.approx(0.1 * (1.0 - 0.8))
    assert rejected_rewards.value == pytest.approx(0.1 * (0.5 - 0.4))


def test_generic_dpo_loss_without_methods() -> None:
    """Verify generic_dpo_loss works on plain scalar numbers lacking detach/mean methods."""
    policy_chosen = 1.0
    policy_rejected = 0.5
    ref_chosen = 0.8
    ref_rejected = 0.4
    beta = 0.1

    def log_sigmoid(x: float) -> float:
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
    state: DPOLoopState[str | None, str | None, dict[str, int]] = DPOLoopState(dataloader=None, epochs=2)
    assert generic_run_training_epochs(state, lambda *_a: 0.0) == 0.0


def test_generic_run_training_epochs_normal() -> None:
    """Verify generic_run_training_epochs averages batch losses with .item()."""
    state: DPOLoopState[str | None, str | None, dict[str, int]] = DPOLoopState(
        dataloader=[{"data": 1}, {"data": 2}],
        epochs=2,
        beta=0.1,
    )

    def step_fn(_policy: str | None, _ref: str | None, _opt: str | None, batch: dict[str, int], _beta: float) -> DummyTensor:
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

    state: DPOLoopState[str | None, str | None, dict[str, int]] = DPOLoopState(
        dataloader=GeneratorLoader(),
        epochs=1,
        beta=0.1,
    )

    def step_fn(_policy: str | None, _ref: str | None, _opt: str | None, batch: dict[str, int], _beta: float) -> float:
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

    def _fail(_m: str, _d: str, _b: float, _ep: int, _lr: float, _bs: int) -> tuple[str, float]:
        """Raise ValueError for testing error handling."""
        raise ValueError("dpo boom")

    res_fail = run_dpo_wrapper("jax", cfg, _fail)
    assert res_fail["status"] == "failed: dpo boom"

    with pytest.raises(TrainingError, match="dpo boom"):
        run_dpo_wrapper("jax", cfg, _fail, raise_on_error=True)

    with pytest.raises(DependencyMissingError, match="missing dpo dep"):
        run_dpo_wrapper("jax", cfg, lambda *_a: (_ for _ in ()).throw(DependencyMissingError("missing dpo dep")))  # pyright: ignore[reportUnknownLambdaType]
