"""Tests for test mlx dpo module."""

import pytest

from gemma_4_sql.backends.mlx import dpo
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import DPOConfig


class MockTensor:
    """Test class for MockTensor."""

    def __init__(self, val=0.0):
        """Initialize __init__."""
        self.val = val

    def mean(self, dim=-1):
        """Execute mean helper."""
        return MockTensor(self.val)

    def __sub__(self, other):
        """Initialize __sub__."""
        return MockTensor(self.val - other.val)

    def __neg__(self):
        """Initialize __neg__."""
        return MockTensor(-self.val)

    def __mul__(self, other):
        """Initialize __mul__."""
        return MockTensor(self.val * other)

    def __rmul__(self, other):
        """Initialize __rmul__."""
        return MockTensor(self.val * other)

    def detach(self):
        """Execute detach helper."""
        return self

    def backward(self):
        """Execute backward helper."""

    def item(self):
        """Execute item helper."""
        return self.val


class MockOptim:
    """Test class for MockOptim."""

    class AdamW:
        """Test class for AdamW."""

        def __init__(self, learning_rate):
            """Initialize __init__."""
            self.learning_rate = learning_rate

        def zero_grad(self):
            """Execute zero grad helper."""

        def step(self):
            """Execute step helper."""


class MockFunctional:
    """Test class for MockFunctional."""

    @staticmethod
    def logsigmoid(x):
        """Execute logsigmoid helper."""
        return MockTensor(-x.val if hasattr(x, "val") else -x)


class MockNN:
    """Test class for MockNN."""

    class losses:
        """Test class for losses."""

        @staticmethod
        def log_sigmoid(x):
            """Execute log sigmoid helper."""
            return MockTensor(x.val if hasattr(x, "val") else x)


class MockMX:
    """Test class for MockMX."""


class MockMLX:
    """Test class for MockMLX."""

    class no_grad:
        """Test class for no grad."""

        def __enter__(self):
            """Initialize __enter__."""

        def __exit__(self, *args):
            """Initialize __exit__."""

    nn = MockNN()


def mock_load(model_name):
    """Execute mock load helper."""

    class MockModel:
        """Test class for MockModel."""

        def __call__(self, x):
            """Initialize __call__."""
            return MockTensor(0.5)

    return MockModel(), None


def test_dpo_loss_missing():
    # When mx is None
    """Test dpo loss missing functionality."""
    dpo.mx = None
    assert dpo.dpo_loss(MockTensor(), MockTensor(), MockTensor(), MockTensor()) == (0.0, 0.0, 0.0)


def test_dpo_loss_present(monkeypatch):
    """Test dpo loss present functionality."""
    dpo.mx = MockMX()
    dpo.mx_nn = MockNN()
    dpo.functional = MockFunctional()
    res = dpo.dpo_loss(MockTensor(1.0), MockTensor(0.5), MockTensor(1.0), MockTensor(0.5), beta=0.1)
    assert len(res) == 3


def test_run_dpo_missing(monkeypatch):
    """Test run dpo missing functionality."""
    dpo.mlx = None
    with pytest.raises(DependencyMissingError):
        dpo.run_dpo(DPOConfig(model_name="m", dataset="d", epochs=1, learning_rate=0.1, beta=0.1))


def test_run_dpo_present(monkeypatch):
    """Test run dpo present functionality."""
    dpo.mlx = MockMLX()
    dpo.nn = MockNN()
    dpo.optim = MockOptim()
    dpo.load = mock_load
    dpo.mx = MockMX()
    dpo.mx_nn = MockNN()
    dpo.functional = MockFunctional()

    # Needs to mock globals() for "load" in globals() check, or we monkeypatch it.
    monkeypatch.setitem(dpo.__dict__, "load", mock_load)

    # Mock build_dataloader to return something iterable
    def mock_build_dataloader(*args, **kwargs):
        """Execute mock build dataloader helper."""
        return {"loader": [{"chosen_inputs": "c", "rejected_inputs": "r"}]}

    monkeypatch.setattr(dpo, "build_dataloader", mock_build_dataloader)

    res = dpo.run_dpo(DPOConfig(model_name="m", dataset="d", epochs=1, learning_rate=0.1, beta=0.1))
    assert res["status"] == "completed"


def test_run_dpo_error(monkeypatch):
    """Test run dpo error functionality."""
    dpo.mlx = MockMLX()
    dpo.nn = MockNN()
    dpo.optim = MockOptim()

    monkeypatch.setitem(dpo.__dict__, "load", mock_load)

    def mock_build_dataloader(*args, **kwargs):
        """Execute mock build dataloader helper."""
        return {"loader": None}  # Causes ValueError

    monkeypatch.setattr(dpo, "build_dataloader", mock_build_dataloader)

    res = dpo.run_dpo(DPOConfig(model_name="m", dataset="d", epochs=1, learning_rate=0.1, beta=0.1))
    assert res["status"].startswith("failed:")


def test_get_train_step_fn_no_methods(monkeypatch):
    """Test _run_dpo_step when optimizer and loss do not have zero_grad, backward, step."""
    dpo.mx = None
    dpo.mx_nn = None
    dpo.functional = None

    class EmptyOpt:
        """Test class for EmptyOpt."""

    class SimpleModel:
        """Test class for SimpleModel."""

        def __call__(self, x):
            """Initialize __call__."""
            return 1.0

    batch = {"chosen_inputs": 1, "rejected_inputs": 2}
    loss = dpo._run_dpo_step(SimpleModel(), SimpleModel(), EmptyOpt(), batch, 0.1)
    assert loss is not None
