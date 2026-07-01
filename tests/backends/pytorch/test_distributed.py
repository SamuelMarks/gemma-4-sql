"""Tests for PyTorch distributed training pipeline."""

from __future__ import annotations

import pytest

import gemma_4_sql.backends.pytorch.export as ex
import gemma_4_sql.backends.pytorch.train as tr
from gemma_4_sql.backends.pytorch import etl


class MockTensor:
    def __init__(self, shape: object, _dtype: object = None, device: object = None) -> None:
        self.shape = shape
        self.device = device

    def to(self, _device: object) -> object:
        return self

    def view(self, *_args: object) -> object:
        return self

    def size(self, *_args: object) -> object:
        return 1

    def backward(self) -> object:
        pass

    def item(self) -> object:
        return 0.1


class MockCuda:
    @staticmethod
    def is_available() -> object:
        return False

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def set_device(_dev: object) -> None:
        pass


class MockDist:
    _init = False

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._init

    @classmethod
    def init_process_group(cls, backend: str) -> None:
        cls._init = True

    @classmethod
    def get_rank(cls) -> int:
        return 0

    @classmethod
    def destroy_process_group(cls) -> None:
        cls._init = False


class MockTorch:
    Tensor = MockTensor
    long = 1
    cuda = MockCuda()
    distributed = MockDist()

    @staticmethod
    def zeros(shape: object, dtype: object = None, device: object = None) -> object:
        return MockTensor(shape, dtype, device)

    @staticmethod
    def device(name: object) -> object:
        return name


class MockDDP:
    def __init__(self, module: object, device_ids: object = None) -> None:
        self.module = module

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.module(*args, **kwargs)

    def train(self) -> None:
        pass

    def parameters(self) -> list[object]:
        return []


class MockFSDP(MockDDP):
    pass


class MockNN:
    @staticmethod
    def CrossEntropyLoss() -> object:
        def loss_fn(*_args: object, **_kwargs: object) -> object:
            return MockTensor((1,))

        return loss_fn

    class parallel:
        DistributedDataParallel = MockDDP


class MockOptim:
    @staticmethod
    def AdamW(*_args: object, **_kwargs: object) -> object:
        class MockOptimizer:
            def zero_grad(self) -> object:
                pass

            def step(self) -> object:
                pass

        return MockOptimizer()


class MockGemma4ForCausalLM:
    @classmethod
    def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
        class MockModel:
            def __call__(self, *_args: object, **_kwargs: object) -> object:
                return MockTensor((1,))

            def to(self, _device: object) -> object:
                return self

            def train(self) -> object:
                pass

            def parameters(self) -> object:
                return []

        return MockModel()


@pytest.fixture
def _mock_torch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setattr(tr, "torch", MockTorch())
    monkeypatch.setattr(tr, "nn", MockNN())
    monkeypatch.setattr(tr, "optim", MockOptim())
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", MockGemma4ForCausalLM)

    # Mock for FSDP which is imported dynamically
    mock_fsdp_module = type("MockFSDPModule", (), {"FullyShardedDataParallel": MockFSDP})
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", mock_fsdp_module)
    monkeypatch.setitem(sys.modules, "torch.nn.parallel", type("MockParallel", (), {"DistributedDataParallel": MockDDP}))

    import torch.distributed

    monkeypatch.setattr(torch.distributed, "is_initialized", MockDist.is_initialized)
    monkeypatch.setattr(torch.distributed, "init_process_group", MockDist.init_process_group)
    monkeypatch.setattr(torch.distributed, "get_rank", MockDist.get_rank)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", MockDist.destroy_process_group)


def test_train_model_ddp(mock_torch_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        return {"loader": [{"inputs": MockTensor((1,)), "targets": MockTensor((1,))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)

    res = tr.train_model("sft", "mod", "dat", 2, 0.1, distributed_strategy="ddp")
    assert res["status"] == "completed"
    assert res["distributed_strategy"] == "ddp"


def test_train_model_fsdp(mock_torch_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        return {"loader": [{"inputs": MockTensor((1,)), "targets": MockTensor((1,))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)

    res = tr.train_model("sft", "mod", "dat", 2, 0.1, distributed_strategy="fsdp")
    assert res["status"] == "completed"
    assert res["distributed_strategy"] == "fsdp"


def test_export_distributed_rank_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    import torch.distributed

    monkeypatch.setattr(ex, "torch", MockTorch())
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    res = ex.export_model("mod", str(tmp_path))
    assert res["status"] == "mock_exported"


def test_export_distributed_rank_one(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    import torch.distributed

    monkeypatch.setattr(ex, "torch", MockTorch())
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    res = ex.export_model("mod", str(tmp_path))
    assert res["status"] == "skipped_non_rank_zero"


def test_etl_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set torch to MockTorch for etl
    monkeypatch.setattr(etl, "torch", MockTorch())

    class MockDataset:
        def __init__(self, *args, **kwargs):
            pass

    class MockDataLoader:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(etl, "Dataset", MockDataset)
    monkeypatch.setattr(etl, "DataLoader", MockDataLoader)
    monkeypatch.setattr(etl, "datasets", type("MockDatasets", (), {"load_dataset": lambda *args, **kwargs: []}))

    import sys

    class MockSampler:
        def __init__(self, *args, **kwargs):
            pass

    mock_dist_sampler = type("MockSamplerModule", (), {"DistributedSampler": MockSampler})
    monkeypatch.setitem(sys.modules, "torch.utils.data.distributed", mock_dist_sampler)

    res = etl.build_dataloader("ds", "train", distributed=True)
    assert res["distributed"] is True
    loader = res["loader"]
    assert hasattr(loader, "kwargs")
    assert loader.kwargs.get("sampler") is not None
