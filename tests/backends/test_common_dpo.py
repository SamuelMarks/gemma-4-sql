"""Unit and behavioral tests for common dpo."""

from typing import Any

import pytest

from gemma_4_sql.backends.common_dpo import generic_dpo_loss, generic_run_training_epochs


class DummyTensor:
    """Test stub for `DummyTensor`."""

    def __init__(self, value: Any, has_detach: Any = True, has_mean: Any = True):
        """Initialize test stub instance."""
        self.value = value
        self.has_detach = has_detach
        self.has_mean = has_mean

    def __sub__(self, other: Any):
        """Provide sub   test helper."""
        return DummyTensor(self.value - other.value, self.has_detach, self.has_mean)

    def __mul__(self, other: Any):
        """Provide mul   test helper."""
        if isinstance(other, DummyTensor):
            return DummyTensor(self.value * other.value, self.has_detach, self.has_mean)
        return DummyTensor(self.value * other, self.has_detach, self.has_mean)

    def __rmul__(self, other: Any):
        """Provide rmul   test helper."""
        return self.__mul__(other)

    def __neg__(self):
        """Provide neg   test helper."""
        return DummyTensor(-self.value, self.has_detach, self.has_mean)

    def item(self):
        """Provide item test helper."""
        return self.value

    def __getattr__(self, name: Any):
        """Provide getattr   test helper."""
        if name == "detach":
            if not self.has_detach:
                raise AttributeError("no detach")
            return lambda: DummyTensor(self.value, self.has_detach, self.has_mean)
        if name == "mean":
            if not self.has_mean:
                raise AttributeError("no mean")
            return lambda: self.value
        raise AttributeError(name)


class TrainerState:
    """Test stub for `TrainerState`."""

    def __init__(self, dataloader: Any, epochs: Any, policy_model: Any, ref_model: Any, optimizer: Any, beta: Any):
        """Initialize test stub instance."""
        self.dataloader = dataloader
        self.epochs = epochs
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.beta = beta


def test_generic_dpo_loss_with_methods():
    """Verify generic dpo loss with methods behavior."""
    policy_chosen: Any = DummyTensor(1.0)
    policy_rejected: Any = DummyTensor(0.5)
    ref_chosen: Any = DummyTensor(0.8)
    ref_rejected: Any = DummyTensor(0.4)
    beta = 0.1

    def log_sigmoid(x: Any) -> Any:
        """Provide log sigmoid test helper."""
        return DummyTensor(x.value * 2)

    loss, chosen_rewards, rejected_rewards = generic_dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, log_sigmoid)
    cr_any: Any = chosen_rewards
    rr_any: Any = rejected_rewards

    assert loss == pytest.approx(-0.02)
    assert cr_any.value == pytest.approx(0.1 * (1.0 - 0.8))
    assert rr_any.value == pytest.approx(0.1 * (0.5 - 0.4))


def test_generic_dpo_loss_without_methods():
    """Verify generic dpo loss without methods behavior."""

    class SimpleTensor:
        """Test stub for `SimpleTensor`."""

        def __init__(self, val: Any):
            """Initialize test stub instance."""
            self.val = val

        def __sub__(self, other: Any):
            """Provide sub   test helper."""
            return SimpleTensor(self.val - other.val)

        def __mul__(self, other: Any):
            """Provide mul   test helper."""
            if isinstance(other, SimpleTensor):
                return SimpleTensor(self.val * other.val)
            return SimpleTensor(self.val * other)

        def __rmul__(self, other: Any):
            """Provide rmul   test helper."""
            return self.__mul__(other)

        def __neg__(self):
            """Provide neg   test helper."""
            return SimpleTensor(-self.val)

    policy_chosen: Any = SimpleTensor(1.0)
    policy_rejected: Any = SimpleTensor(0.5)
    ref_chosen: Any = SimpleTensor(0.8)
    ref_rejected: Any = SimpleTensor(0.4)
    beta = 0.1

    def log_sigmoid(x: Any) -> Any:
        """Provide log sigmoid test helper."""
        return SimpleTensor(x.val * 2)

    # Test SimpleTensor * SimpleTensor (line 109)
    t1 = SimpleTensor(2.0)
    t2 = SimpleTensor(3.0)
    t3 = t1 * t2
    assert t3.val == 6.0

    loss, chosen_rewards, rejected_rewards = generic_dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, log_sigmoid)
    loss_any: Any = loss
    cr_any: Any = chosen_rewards
    rr_any: Any = rejected_rewards

    assert loss_any.val == pytest.approx(-0.02)
    assert cr_any.val == pytest.approx(0.02)
    assert rr_any.val == pytest.approx(0.01)


def test_generic_run_training_epochs_none_dataloader():
    """Verify generic run training epochs none dataloader behavior."""
    state: Any = TrainerState(None, 2, None, None, None, 0.1)

    def step_fn(*args: Any):
        """Provide step fn test helper."""

    assert generic_run_training_epochs(state, step_fn) == 0.0


def test_generic_run_training_epochs_normal():
    """Verify generic run training epochs normal behavior."""
    state: Any = TrainerState([{"data": 1}, {"data": 2}], 2, None, None, None, 0.1)

    def step_fn(policy: Any, ref: Any, opt: Any, batch: Any, beta: Any):
        """Provide step fn test helper."""
        return DummyTensor(batch["data"] * 0.5)

    final_loss = generic_run_training_epochs(state, step_fn)
    assert final_loss == 0.75


def test_generic_run_training_epochs_no_item():
    """Verify generic run training epochs no item behavior."""
    state: Any = TrainerState([{"data": 1}, {"data": 2}], 1, None, None, None, 0.1)

    def step_fn(policy: Any, ref: Any, opt: Any, batch: Any, beta: Any):
        """Provide step fn test helper."""
        return batch["data"] * 0.5

    final_loss = generic_run_training_epochs(state, step_fn)
    assert final_loss == 0.75


def test_generic_run_training_epochs_no_len_dataloader():
    """Verify generic run training epochs no len dataloader behavior."""

    class GeneratorMock:
        """Test stub for `GeneratorMock`."""

        def __iter__(self):
            """Provide iter   test helper."""
            yield {"data": 1}
            yield {"data": 2}

    state: Any = TrainerState(GeneratorMock(), 1, None, None, None, 0.1)

    def step_fn(policy: Any, ref: Any, opt: Any, batch: Any, beta: Any):
        """Provide step fn test helper."""
        return batch["data"] * 0.5

    final_loss = generic_run_training_epochs(state, step_fn)
    assert final_loss == 0.75


def test_dummy_tensor_edge_cases():
    """Test edge cases in DummyTensor."""
    # Line 24: DummyTensor * DummyTensor
    t1 = DummyTensor(2.0)
    t2 = DummyTensor(3.0)
    t3 = t1 * t2
    assert t3.value == 6.0

    # Line 38 & 44: detach and mean when flags are false
    t_no_detach = DummyTensor(1.0, has_detach=False)
    with pytest.raises(AttributeError, match="no detach"):
        t_no_detach.detach()

    t_no_mean = DummyTensor(1.0, has_mean=False)
    with pytest.raises(AttributeError, match="no mean"):
        t_no_mean.mean()

    # Lines 53-57: getattr fallbacks called explicitly
    with pytest.raises(AttributeError):
        _ = t_no_detach.detach

    with pytest.raises(AttributeError):
        _ = t_no_mean.mean

    with pytest.raises(AttributeError):
        _ = t1.nonexistent_attr
