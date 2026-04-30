"""Tests for PyTorch training pipeline."""

import pytest

import gemma_4_sql.backends.pytorch.train as tr
from gemma_4_sql.backends.pytorch.train import train_model


class MockTensor:
    def __init__(self, shape, dtype=None, device=None):
        self.shape = shape
        self.device = device

    def to(self, device):
        return self

    def view(self, *args):
        return self

    def size(self, *args):
        return 1

    def backward(self):
        pass

    def item(self):
        return 0.1


class MockCuda:
    @staticmethod
    def is_available():
        return False


class MockTorch:
    Tensor = MockTensor
    long = 1
    cuda = MockCuda()

    @staticmethod
    def zeros(shape, dtype=None, device=None):
        return MockTensor(shape, dtype, device)

    @staticmethod
    def device(name):
        return name


class MockNN:
    @staticmethod
    def CrossEntropyLoss():
        def loss_fn(*args, **kwargs):
            return MockTensor((1,))

        return loss_fn


class MockOptim:
    @staticmethod
    def AdamW(*args, **kwargs):
        class MockOptimizer:
            def zero_grad(self):
                pass

            def step(self):
                pass

        return MockOptimizer()


class MockGemma4ForCausalLM:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        class MockModel:
            def __call__(self, *args, **kwargs):
                return MockTensor((1,))

            def to(self, device):
                return self

            def train(self):
                pass

            def parameters(self):
                return []

        return MockModel()


@pytest.fixture
def mock_torch_env(monkeypatch):
    monkeypatch.setattr(tr, "torch", MockTorch())
    monkeypatch.setattr(tr, "nn", MockNN())
    monkeypatch.setattr(tr, "optim", MockOptim())
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", MockGemma4ForCausalLM)

    def mock_build_dataloader(*args, **kwargs):
        return {"loader": [{"inputs": MockTensor((1,)), "targets": MockTensor((1,))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)


def test_train_model_pytorch_real(mock_torch_env):
    res = train_model("sft", "mod", "dat", 2, 0.1)
    assert res["status"] == "completed"
    # the loop computes loss using batch item
    assert res["final_loss"] == 0.1
    assert res["backend"] == "pytorch"


def test_train_model_pytorch_missing():
    orig_torch = tr.torch
    tr.torch = None
    res = train_model("sft", "mod", "dat", 2, 0.1)
    assert res["status"] == "mocked_missing_torch"
    tr.torch = orig_torch


def test_train_model_pytorch_error(mock_torch_env, monkeypatch):
    def raise_error(*args, **kwargs):
        raise ValueError("err")

    monkeypatch.setattr(tr, "build_dataloader", raise_error)
    res = train_model("sft", "mod", "dat", 2, 0.1)
    assert "failed: err" in res["status"]


def test_train_model_pytorch_no_loader_fallback(mock_torch_env, monkeypatch):
    def mock_build_dataloader(*args, **kwargs):
        return {"loader": None}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    res = train_model("sft", "mod", "dat", 2, 0.1)
    assert res["status"] == "completed"
    assert res["final_loss"] == 0.35
