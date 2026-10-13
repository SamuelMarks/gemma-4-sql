"""Module docstring."""

import pytest

from gemma_4_sql.backends.common_dpo import generic_dpo_loss, generic_run_training_epochs


class DummyTensor:
    """Docstring for DummyTensor."""

    def __init__(self, value, has_detach=True, has_mean=True):
        """Docstring for __init__."""
        self.value = value
        self.has_detach = has_detach
        self.has_mean = has_mean

    def __sub__(self, other):
        """Docstring for __sub__."""
        return DummyTensor(self.value - other.value, self.has_detach, self.has_mean)

    def __mul__(self, other):
        """Docstring for __mul__."""
        if isinstance(other, DummyTensor):
            return DummyTensor(self.value * other.value, self.has_detach, self.has_mean)
        return DummyTensor(self.value * other, self.has_detach, self.has_mean)

    def __rmul__(self, other):
        """Docstring for __rmul__."""
        return self.__mul__(other)

    def __neg__(self):
        """Docstring for __neg__."""
        return DummyTensor(-self.value, self.has_detach, self.has_mean)

    def item(self):
        """Docstring for item."""
        return self.value

    def __getattr__(self, name):
        """Docstring for __getattr__."""
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
    """Docstring for TrainerState."""

    def __init__(self, dataloader, epochs, policy_model, ref_model, optimizer, beta):
        """Docstring for __init__."""
        self.dataloader = dataloader
        self.epochs = epochs
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.beta = beta


def test_generic_dpo_loss_with_methods():
    """Docstring for test_generic_dpo_loss_with_methods."""
    policy_chosen = DummyTensor(1.0)
    policy_rejected = DummyTensor(0.5)
    ref_chosen = DummyTensor(0.8)
    ref_rejected = DummyTensor(0.4)
    beta = 0.1

    def log_sigmoid(x):
        """Docstring for log_sigmoid."""
        return DummyTensor(x.value * 2)

    loss, chosen_rewards, rejected_rewards = generic_dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, log_sigmoid)

    assert loss == pytest.approx(-0.02)
    assert chosen_rewards.value == pytest.approx(0.1 * (1.0 - 0.8))
    assert rejected_rewards.value == pytest.approx(0.1 * (0.5 - 0.4))


def test_generic_dpo_loss_without_methods():
    """Docstring for test_generic_dpo_loss_without_methods."""

    class SimpleTensor:
        """Docstring for SimpleTensor."""

        def __init__(self, val):
            """Docstring for __init__."""
            self.val = val

        def __sub__(self, other):
            """Docstring for __sub__."""
            return SimpleTensor(self.val - other.val)

        def __mul__(self, other):
            """Docstring for __mul__."""
            if isinstance(other, SimpleTensor):
                return SimpleTensor(self.val * other.val)
            return SimpleTensor(self.val * other)

        def __rmul__(self, other):
            """Docstring for __rmul__."""
            return self.__mul__(other)

        def __neg__(self):
            """Docstring for __neg__."""
            return SimpleTensor(-self.val)

    policy_chosen = SimpleTensor(1.0)
    policy_rejected = SimpleTensor(0.5)
    ref_chosen = SimpleTensor(0.8)
    ref_rejected = SimpleTensor(0.4)
    beta = 0.1

    def log_sigmoid(x):
        """Docstring for log_sigmoid."""
        return SimpleTensor(x.val * 2)

    # Test SimpleTensor * SimpleTensor (line 109)
    t1 = SimpleTensor(2.0)
    t2 = SimpleTensor(3.0)
    t3 = t1 * t2
    assert t3.val == 6.0

    loss, chosen_rewards, rejected_rewards = generic_dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, log_sigmoid)

    assert loss.val == pytest.approx(-0.02)
    assert chosen_rewards.val == pytest.approx(0.02)
    assert rejected_rewards.val == pytest.approx(0.01)


def test_generic_run_training_epochs_none_dataloader():
    """Docstring for test_generic_run_training_epochs_none_dataloader."""
    state = TrainerState(None, 2, None, None, None, 0.1)

    def step_fn(*args):
        """Docstring for step_fn."""

    assert generic_run_training_epochs(state, step_fn) == 0.0


def test_generic_run_training_epochs_normal():
    """Docstring for test_generic_run_training_epochs_normal."""
    state = TrainerState([{"data": 1}, {"data": 2}], 2, None, None, None, 0.1)

    def step_fn(policy, ref, opt, batch, beta):
        """Docstring for step_fn."""
        return DummyTensor(batch["data"] * 0.5)

    final_loss = generic_run_training_epochs(state, step_fn)
    assert final_loss == 0.75


def test_generic_run_training_epochs_no_item():
    """Docstring for test_generic_run_training_epochs_no_item."""
    state = TrainerState([{"data": 1}, {"data": 2}], 1, None, None, None, 0.1)

    def step_fn(policy, ref, opt, batch, beta):
        """Docstring for step_fn."""
        return batch["data"] * 0.5

    final_loss = generic_run_training_epochs(state, step_fn)
    assert final_loss == 0.75


def test_generic_run_training_epochs_no_len_dataloader():
    """Docstring for test_generic_run_training_epochs_no_len_dataloader."""

    class GeneratorMock:
        """Docstring for GeneratorMock."""

        def __iter__(self):
            """Docstring for __iter__."""
            yield {"data": 1}
            yield {"data": 2}

    state = TrainerState(GeneratorMock(), 1, None, None, None, 0.1)

    def step_fn(policy, ref, opt, batch, beta):
        """Docstring for step_fn."""
        return batch["data"] * 0.5

    final_loss = generic_run_training_epochs(state, step_fn)
    assert final_loss == 1.5


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
