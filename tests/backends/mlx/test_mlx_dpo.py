import pytest

from gemma_4_sql.backends.mlx import dpo
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import DPOConfig


class MockTensor:
    def __init__(self, val=0.0):
        self.val = val

    def mean(self, dim=-1):
        return MockTensor(self.val)

    def __sub__(self, other):
        return MockTensor(self.val - other.val)

    def __neg__(self):
        return MockTensor(-self.val)

    def __mul__(self, other):
        return MockTensor(self.val * other)

    def __rmul__(self, other):
        return MockTensor(self.val * other)

    def detach(self):
        return self

    def backward(self):
        pass

    def item(self):
        return self.val


class MockOptim:
    class AdamW:
        def __init__(self, learning_rate):
            self.learning_rate = learning_rate

        def zero_grad(self):
            pass

        def step(self):
            pass


class MockFunctional:
    @staticmethod
    def logsigmoid(x):
        return MockTensor(-x.val if hasattr(x, "val") else -x)


class MockNN:
    class losses:
        @staticmethod
        def log_sigmoid(x):
            return MockTensor(x.val if hasattr(x, "val") else x)


class MockMX:
    pass


class MockMLX:
    class no_grad:
        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

    nn = MockNN()


def mock_load(model_name):
    class MockModel:
        def __call__(self, x):
            return MockTensor(0.5)

    return MockModel(), None


def test_dpo_loss_missing():
    # When mx is None
    dpo.mx = None
    assert dpo.dpo_loss(MockTensor(), MockTensor(), MockTensor(), MockTensor()) == (0.0, 0.0, 0.0)


def test_dpo_loss_present(monkeypatch):
    dpo.mx = MockMX()
    dpo.mx_nn = MockNN()
    dpo.functional = MockFunctional()
    res = dpo.dpo_loss(MockTensor(1.0), MockTensor(0.5), MockTensor(1.0), MockTensor(0.5), beta=0.1)
    assert len(res) == 3


def test_run_dpo_missing(monkeypatch):
    dpo.mlx = None
    with pytest.raises(DependencyMissingError):
        dpo.run_dpo(DPOConfig(model_name="m", dataset="d", epochs=1, learning_rate=0.1, beta=0.1))


def test_run_dpo_present(monkeypatch):
    dpo.mlx = MockMLX()
    dpo.nn = MockNN()
    dpo.optim = MockOptim()
    dpo.load = mock_load
    dpo.mx = MockMX()
    dpo.mx_nn = MockNN()

    # Needs to mock globals() for "load" in globals() check, or we monkeypatch it.
    monkeypatch.setitem(dpo.__dict__, "load", mock_load)

    # Mock build_dataloader to return something iterable
    def mock_build_dataloader(*args, **kwargs):
        return {"loader": [{"chosen_inputs": "c", "rejected_inputs": "r"}]}

    monkeypatch.setattr(dpo, "build_dataloader", mock_build_dataloader)

    res = dpo.run_dpo(DPOConfig(model_name="m", dataset="d", epochs=1, learning_rate=0.1, beta=0.1))
    assert res["status"] == "completed"


def test_run_dpo_error(monkeypatch):
    dpo.mlx = MockMLX()
    dpo.nn = MockNN()
    dpo.optim = MockOptim()

    monkeypatch.setitem(dpo.__dict__, "load", mock_load)

    def mock_build_dataloader(*args, **kwargs):
        return {"loader": None}  # Causes ValueError

    monkeypatch.setattr(dpo, "build_dataloader", mock_build_dataloader)

    res = dpo.run_dpo(DPOConfig(model_name="m", dataset="d", epochs=1, learning_rate=0.1, beta=0.1))
    assert res["status"].startswith("failed:")
