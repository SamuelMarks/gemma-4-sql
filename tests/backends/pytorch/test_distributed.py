# Copyright 2024
"""Tests for PyTorch distributed training pipeline."""

from __future__ import annotations

import typing

import pytest

import gemma_4_sql.backends.pytorch.export as ex
import gemma_4_sql.backends.pytorch.train as tr
from gemma_4_sql.backends.pytorch import etl
from gemma_4_sql.type_hints import ETLConfig, TrainingConfig


class MockTensor:
    """Provide class docstring."""

    def __init__(self, shape: object, _dtype: object = None, device: object = None) -> None:
        """Execute function."""
        self.shape = shape
        self.device = device

    def to(self, _device: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self

    def view(self, *_args: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self

    def size(self, *_args: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return 1

    def backward(self) -> object:
        """Execute function."""

    def item(self) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return 0.1


class MockCuda:
    """Provide class docstring."""

    @staticmethod
    def is_available() -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return False

    @staticmethod
    def device_count() -> int:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return 1

    @staticmethod
    def set_device(_dev: object) -> None:
        """Execute function."""


class MockDist:
    """Provide class docstring."""

    _init = False

    @classmethod
    def is_initialized(cls) -> bool:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return cls._init

    @classmethod
    def init_process_group(cls, _backend: str) -> None:
        """Execute function."""
        cls._init = True

    @classmethod
    def get_rank(cls) -> int:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return 0

    @classmethod
    def destroy_process_group(cls) -> None:
        """Execute function."""
        cls._init = False


class MockTorch:
    """Provide class docstring."""

    Tensor = MockTensor
    long = 1
    cuda = MockCuda()
    distributed = MockDist()

    @staticmethod
    def zeros(shape: object, dtype: object = None, device: object = None) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockTensor(shape, dtype, device)

    @staticmethod
    def device(name: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return name


class MockDDP:
    """Provide class docstring."""

    def __init__(self, module: object, device_ids: object = None) -> None:
        """Execute function."""
        self.module = module

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self.module(*args, **kwargs)

    def train(self) -> None:
        """Execute function."""

    def parameters(self) -> list[typing.Any]:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return []


class MockFSDP(MockDDP):
    """Provide class docstring."""


class MockNN:
    """Provide class docstring."""

    @staticmethod
    def mock_crossentropyloss() -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """

        def loss_fn(*_args: object, **_kwargs: object) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return MockTensor((1,))

        return loss_fn

    CrossEntropyLoss = mock_crossentropyloss

    class MockParallel:
        """Provide class docstring."""

        DistributedDataParallel = MockDDP

    parallel = MockParallel


class MockOptim:
    """Provide class docstring."""

    @staticmethod
    def mock_adamw(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """

        class MockOptimizer:
            """Provide class docstring."""

            def zero_grad(self) -> object:
                """Execute function."""

            def step(self) -> object:
                """Execute function."""

        return MockOptimizer()

    AdamW = mock_adamw


class MockGemma4ForCausalLM:
    """Provide class docstring."""

    @classmethod
    def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """

        class MockModel:
            """Provide class docstring."""

            def __call__(self, *_args: object, **_kwargs: object) -> object:
                """Execute function.

                Returns:
                    object: Description of return.

                """
                return MockTensor((1,))

            def to(self, _device: object) -> object:
                """Execute function.

                Returns:
                    object: Description of return.

                """
                return self

            def train(self) -> object:
                """Execute function."""

            def parameters(self) -> object:
                """Execute function.

                Returns:
                    object: Description of return.

                """
                return []

        return MockModel()


@pytest.fixture
def _mock_torch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    sys = __import__("sys")

    monkeypatch.setattr(tr, "torch", MockTorch())
    monkeypatch.setattr(tr, "nn", MockNN())
    monkeypatch.setattr(tr, "optim", MockOptim())
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    mock_fsdp_module = type("MockFSDPModule", (), {"FullyShardedDataParallel": MockFSDP})
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", mock_fsdp_module)
    monkeypatch.setitem(sys.modules, "torch.nn.parallel", type("MockParallel", (), {"DistributedDataParallel": MockDDP}))
    torch = __import__("torch.distributed")

    monkeypatch.setattr(torch.distributed, "is_initialized", MockDist.is_initialized)
    monkeypatch.setattr(torch.distributed, "init_process_group", MockDist.init_process_group)
    monkeypatch.setattr(torch.distributed, "get_rank", MockDist.get_rank)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", MockDist.destroy_process_group)


@pytest.mark.usefixtures("_mock_torch_env")
def test_train_model_ddp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return {"loader": [{"inputs": MockTensor((1,)), "targets": MockTensor((1,))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    res = tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1, distributed_strategy="ddp"))
    assert res["status"] == "completed"
    if res["distributed_strategy"] != "ddp":
        raise AssertionError


@pytest.mark.usefixtures("_mock_torch_env")
def test_train_model_fsdp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return {"loader": [{"inputs": MockTensor((1,)), "targets": MockTensor((1,))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    res = tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1, distributed_strategy="fsdp"))
    assert res["status"] == "completed"
    if res["distributed_strategy"] != "fsdp":
        raise AssertionError


def test_export_distributed_rank_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    torch = __import__("torch.distributed")

    monkeypatch.setattr(ex, "torch", MockTorch())
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    res = ex.export_model("mod", str(tmp_path))
    if res["status"] != "mock_exported":
        raise AssertionError


def test_export_distributed_rank_one(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    torch = __import__("torch.distributed")

    monkeypatch.setattr(ex, "torch", MockTorch())
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    res = ex.export_model("mod", str(tmp_path))
    if res["status"] != "skipped_non_rank_zero":
        raise AssertionError


def test_etl_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(etl, "torch", MockTorch())

    class MockDataset:
        """Provide class docstring."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Execute function."""

    class MockDataLoader:
        """Provide class docstring."""

        def __init__(self, *_args: object, **kwargs: object) -> None:
            """Execute function."""
            self.kwargs = kwargs

    monkeypatch.setattr(etl, "Dataset", MockDataset)
    monkeypatch.setattr(etl, "DataLoader", MockDataLoader)
    monkeypatch.setattr(etl, "datasets", type("MockDatasets", (), {"load_dataset": lambda *_args, **_kwargs: []}))
    sys = __import__("sys")

    class MockSampler:
        """Provide class docstring."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Execute function."""

    mock_dist_sampler = type("MockSamplerModule", (), {"DistributedSampler": MockSampler})
    monkeypatch.setitem(sys.modules, "torch.utils.data.distributed", mock_dist_sampler)
    res = etl.build_dataloader(ETLConfig(dataset_name="ds", split="train", distributed=True))
    if res["distributed"] is not True:
        raise AssertionError
    loader = res["loader"]
    if not (hasattr(loader, "kwargs")):
        raise AssertionError
    if loader.kwargs.get("sampler") is None:
        raise AssertionError
