import pytest
import torch
from torch import nn

import gemma_4_sql.backends.pytorch.train as tr
from gemma_4_sql.backends.pytorch.train import _cleanup_distributed, _execute_train, _setup_distributed, _wrap_model_distributed, train_model
from gemma_4_sql.type_hints import TrainingConfig


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 16)
        self.linear = nn.Linear(16, 100)

    def forward(self, x):
        x = self.embed(x)
        return self.linear(x)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


@pytest.fixture
def mock_transformers_gemma(monkeypatch):
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", DummyModel)


@pytest.fixture
def mock_build_dataloader(monkeypatch):
    def mock_build(*args, **kwargs):
        return {"loader": [{"inputs": torch.randint(0, 100, (2, 10)), "targets": torch.randint(0, 100, (2, 10))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build)


def test_train_model_pytorch_real(mock_transformers_gemma, mock_build_dataloader):
    config = TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1)
    res = train_model(config)
    assert res["backend"] == "pytorch"
    assert res["status"] == "completed"


def test_train_model_pytorch_missing(monkeypatch):
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(tr, "torch", None)
    with pytest.raises(DependencyMissingError, match="PyTorch dependencies are missing"):
        train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1))


def test_execute_train_missing_deps(monkeypatch):
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(tr, "torch", None)
    with pytest.raises(DependencyMissingError, match="PyTorch dependencies are missing"):
        _execute_train("mod", "ds", 1, 1e-4, "none")


def test_train_model_pytorch_error(mock_transformers_gemma, monkeypatch):
    monkeypatch.setattr(tr, "build_dataloader", lambda *a, **k: Exception("err"))
    config = TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1)
    res = train_model(config)
    assert "failed" in res["status"]


def test_train_model_pytorch_no_loader_fallback(mock_transformers_gemma, monkeypatch):
    monkeypatch.setattr(tr, "build_dataloader", lambda *a, **k: {"loader": None})
    config = TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1)
    res = train_model(config)
    assert "failed" in res["status"]


def test_execute_train_success(mock_transformers_gemma, monkeypatch):
    def mock_build(*args, **kwargs):
        return {"loader": [{"inputs": torch.randint(0, 100, (2, 10)), "targets": torch.randint(0, 100, (2, 10))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build)
    res = tr._execute_train("test_ds", "ds", 1, 1e-4, distributed_strategy="none", batch_size=2)
    assert res[0] == "completed"
    assert isinstance(res[1], float)


def test_execute_train_loss_tuple(mock_transformers_gemma, monkeypatch):
    class TupleModel:
        def __call__(self, x):
            t = torch.randn(2, 10, 100)
            t.requires_grad = True
            return (t,)

        def to(self, d):
            return self

        def parameters(self):
            return [torch.nn.Parameter(torch.randn(1))]

        def train(self, mode=True):
            return self

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

    monkeypatch.setattr(tr, "Gemma4ForCausalLM", TupleModel)

    def mock_build(*args, **kwargs):
        return {"loader": [{"inputs": torch.randint(0, 100, (2, 10)), "targets": torch.randint(0, 100, (2, 10))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build)
    res = tr._execute_train("test_ds", "ds", 1, 1e-4, distributed_strategy="none", batch_size=2)
    assert res[0] == "completed"


def test_pytorch_setup_distributed(monkeypatch):
    is_dist, d, _device, _rank = _setup_distributed("none")
    assert not is_dist
    assert d is None

    # Test ddp branch with mocks since we don't have multiple GPUs
    class MockDist:
        @staticmethod
        def is_initialized():
            return False

        @staticmethod
        def init_process_group(*a):
            pass

        @staticmethod
        def get_rank():
            return 0

        @staticmethod
        def destroy_process_group():
            pass

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "torch.distributed":
            return MockDist
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    is_dist, d, _device, _rank = _setup_distributed("ddp")
    assert is_dist
    assert d is MockDist

    _cleanup_distributed(d)

    class MockInitializedDist:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def destroy_process_group():
            pass

    _cleanup_distributed(MockInitializedDist)


def test_wrap_model_distributed():
    model = nn.Linear(10, 10)
    # Testing "none"
    wrapped = _wrap_model_distributed(model, "none", 0)
    assert wrapped is model

    # We can't easily test DDP/FSDP without initialized process groups,
    # but we can monkeypatch importlib to return mock modules


def test_wrap_model_distributed_real(monkeypatch):
    model = nn.Linear(10, 10)

    # DDP mock
    import importlib

    orig_import_module = importlib.import_module

    def mock_import_module(name):
        if name == "torch.nn.parallel":
            return type("MockDDPModule", (), {"DistributedDataParallel": lambda m, **kwargs: m})
        if name == "torch.distributed.fsdp":
            return type("MockFSDPModule", (), {"FullyShardedDataParallel": lambda m, **kwargs: m})
        return orig_import_module(name)

    monkeypatch.setattr("importlib.import_module", mock_import_module)

    wrapped_ddp = _wrap_model_distributed(model, "ddp", 0)
    assert wrapped_ddp is model

    wrapped_fsdp = _wrap_model_distributed(model, "fsdp", 0)
    assert wrapped_fsdp is model


def test_train_model_pytorch_native(mock_build_dataloader, monkeypatch):
    class MockNativeGemma:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return DummyModel()

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM", MockNativeGemma, raising=False)

    config = TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=1, learning_rate=0.1, backend="pytorch_native")
    res = train_model(config)
    assert res["status"] == "completed"


def test_train_pytorch_import_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    mock_gemma4 = type("gemma4", (), {"Gemma4ForCausalLM": "mocked"})
    monkeypatch.setitem(sys.modules, "transformers.models.gemma4", mock_gemma4)
    import gemma_4_sql.backends.pytorch.train as tr

    importlib.reload(tr)
    assert tr.Gemma4ForCausalLM == "mocked"
    del sys.modules["transformers.models.gemma4"]
    importlib.reload(tr)


def test_setup_distributed_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from gemma_4_sql.backends.pytorch.train import _setup_distributed

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: None)

    class MockDist:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_rank():
            return 0

    import sys

    monkeypatch.setitem(sys.modules, "torch.distributed", MockDist)

    is_dist, _d, dev, _rank = _setup_distributed("ddp")
    assert is_dist
    assert dev.type == "cuda"
