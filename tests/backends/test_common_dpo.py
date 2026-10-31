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
    assert float(1.0 - policy_chosen) == pytest.approx(0.0)
    assert chosen_rewards.value == pytest.approx(0.1 * (1.0 - 0.8))
    assert rejected_rewards.value == pytest.approx(0.1 * (0.5 - 0.4))


def test_generic_dpo_loss_without_methods() -> None:
    """Verify generic_dpo_loss works on ArithmeticTensorProtocol objects lacking detach/mean methods."""

    class PlainArith:
        """Minimal arithmetic wrapper without `.detach()` or `.mean()`."""

        def __init__(self, val: float) -> None:
            """Initialize wrapper."""
            self.val = val

        def __float__(self) -> float:
            """Convert to float."""
            return self.val

        def __sub__(self, other: PlainArith | float, /) -> PlainArith:
            """Subtract."""
            return PlainArith(self.val - float(other))

        def __rsub__(self, other: PlainArith | float, /) -> PlainArith:
            """Right subtract."""
            return PlainArith(float(other) - self.val)

        def __mul__(self, other: PlainArith | float, /) -> PlainArith:
            """Multiply."""
            return PlainArith(self.val * float(other))

        def __rmul__(self, other: PlainArith | float, /) -> PlainArith:
            """Right multiply."""
            return PlainArith(float(other) * self.val)

        def __neg__(self) -> PlainArith:
            """Negate."""
            return PlainArith(-self.val)

    policy_chosen = PlainArith(1.0)
    policy_rejected = PlainArith(0.5)
    ref_chosen = PlainArith(0.8)
    ref_rejected = PlainArith(0.4)
    beta = 0.1

    def log_sigmoid(x: PlainArith) -> PlainArith:
        """Scale value by 2."""
        return PlainArith(x.val * 2.0)

    loss, chosen_rewards, rejected_rewards = generic_dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta,
        log_sigmoid,
    )

    assert float(loss) == pytest.approx(-0.02)
    assert float(1.0 - policy_chosen) == pytest.approx(0.0)
    assert float(policy_chosen * 2.0) == pytest.approx(2.0)
    assert chosen_rewards.val == pytest.approx(0.02)
    assert rejected_rewards.val == pytest.approx(0.01)

    def log_sigmoid_float(x: float) -> float:
        """Scale float by 2."""
        return float(x) * 2.0

    loss_f, ch_f, rej_f = generic_dpo_loss(1.0, 0.5, 0.8, 0.4, 0.1, log_sigmoid_float)
    assert loss_f == pytest.approx(-0.02)
    assert ch_f == pytest.approx(0.02)
    assert rej_f == pytest.approx(0.01)


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

    def _raise_dep(_m: str, _d: str, _b: float, _ep: int, _lr: float, _bs: int) -> tuple[str, float]:
        """Raise DependencyMissingError."""
        raise DependencyMissingError("missing dpo dep")

    with pytest.raises(DependencyMissingError, match="missing dpo dep"):
        run_dpo_wrapper("jax", cfg, _raise_dep)
