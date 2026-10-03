import pytest

from gemma_4_sql.backends.common_dpo import generic_dpo_loss, generic_run_training_epochs


class DummyTensor:
    def __init__(self, value, has_detach=True, has_mean=True):
        self.value = value
        self.has_detach = has_detach
        self.has_mean = has_mean

    def __sub__(self, other):
        return DummyTensor(self.value - other.value, self.has_detach, self.has_mean)

    def __mul__(self, other):
        if isinstance(other, DummyTensor):
            return DummyTensor(self.value * other.value, self.has_detach, self.has_mean)
        return DummyTensor(self.value * other, self.has_detach, self.has_mean)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return DummyTensor(-self.value, self.has_detach, self.has_mean)

    def detach(self):
        if not self.has_detach:
            raise AttributeError("no detach")
        return DummyTensor(self.value, self.has_detach, self.has_mean)

    def mean(self):
        if not self.has_mean:
            raise AttributeError("no mean")
        return self.value

    def item(self):
        return self.value

    def __getattr__(self, name):
        if name == "detach" and not self.has_detach:
            raise AttributeError
        if name == "mean" and not self.has_mean:
            raise AttributeError
        raise AttributeError(name)


class TrainerState:
    def __init__(self, dataloader, epochs, policy_model, ref_model, optimizer, beta):
        self.dataloader = dataloader
        self.epochs = epochs
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.beta = beta


def test_generic_dpo_loss_with_methods():
    policy_chosen = DummyTensor(1.0)
    policy_rejected = DummyTensor(0.5)
    ref_chosen = DummyTensor(0.8)
    ref_rejected = DummyTensor(0.4)
    beta = 0.1

    def log_sigmoid(x):
        return DummyTensor(x.value * 2)

    loss, chosen_rewards, rejected_rewards = generic_dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, log_sigmoid)

    assert loss == pytest.approx(-0.02)
    assert chosen_rewards.value == pytest.approx(0.1 * (1.0 - 0.8))
    assert rejected_rewards.value == pytest.approx(0.1 * (0.5 - 0.4))


def test_generic_dpo_loss_without_methods():
    class SimpleTensor:
        def __init__(self, val):
            self.val = val

        def __sub__(self, other):
            return SimpleTensor(self.val - other.val)

        def __mul__(self, other):
            if isinstance(other, SimpleTensor):
                return SimpleTensor(self.val * other.val)
            return SimpleTensor(self.val * other)

        def __rmul__(self, other):
            return self.__mul__(other)

        def __neg__(self):
            return SimpleTensor(-self.val)

    policy_chosen = SimpleTensor(1.0)
    policy_rejected = SimpleTensor(0.5)
    ref_chosen = SimpleTensor(0.8)
    ref_rejected = SimpleTensor(0.4)
    beta = 0.1

    def log_sigmoid(x):
        return SimpleTensor(x.val * 2)

    loss, chosen_rewards, rejected_rewards = generic_dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, log_sigmoid)

    assert loss.val == pytest.approx(-0.02)
    assert chosen_rewards.val == pytest.approx(0.02)
    assert rejected_rewards.val == pytest.approx(0.01)


def test_generic_run_training_epochs_none_dataloader():
    state = TrainerState(None, 2, None, None, None, 0.1)

    def step_fn(*args):
        pass

    assert generic_run_training_epochs(state, step_fn) == 0.0


def test_generic_run_training_epochs_normal():
    state = TrainerState([{"data": 1}, {"data": 2}], 2, None, None, None, 0.1)

    def step_fn(policy, ref, opt, batch, beta):
        return DummyTensor(batch["data"] * 0.5)

    final_loss = generic_run_training_epochs(state, step_fn)
    assert final_loss == 0.75


def test_generic_run_training_epochs_no_item():
    state = TrainerState([{"data": 1}, {"data": 2}], 1, None, None, None, 0.1)

    def step_fn(policy, ref, opt, batch, beta):
        return batch["data"] * 0.5

    final_loss = generic_run_training_epochs(state, step_fn)
    assert final_loss == 0.75


def test_generic_run_training_epochs_no_len_dataloader():
    class GeneratorMock:
        def __iter__(self):
            yield {"data": 1}
            yield {"data": 2}

    state = TrainerState(GeneratorMock(), 1, None, None, None, 0.1)

    def step_fn(policy, ref, opt, batch, beta):
        return batch["data"] * 0.5

    final_loss = generic_run_training_epochs(state, step_fn)
    assert final_loss == 1.5
