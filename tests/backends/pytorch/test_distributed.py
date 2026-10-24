"""Tests for PyTorch distributed training and export pipelines."""

from __future__ import annotations

import os
import sys
import typing
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from tests.conftest import requires_torch

pytestmark = requires_torch

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import gemma_4_sql.backends.pytorch.export as ex
import gemma_4_sql.backends.pytorch.train as tr
from gemma_4_sql.backends.pytorch import etl
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import ETLConfig, TrainingConfig


def mock_build_dataloader(config: ETLConfig, *_args: Any, **_kwargs: Any) -> dict[str, typing.Any]:
    """Build a synthetic PyTorch DataLoader with optional DistributedSampler."""
    inputs = torch.randint(0, 128, (4, 10))
    targets = torch.randint(0, 128, (4, 10))

    class DictDataset(torch.utils.data.Dataset[dict[str, Any]]):
        """Dataset yielding input/target tensor dictionaries."""

        def __init__(self, inp: Any, tgt: Any) -> None:
            """Store input and target tensors."""
            self.inputs = inp
            self.targets = tgt

        def __len__(self) -> int:
            """Return dataset length."""
            return len(self.inputs)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            """Return single sample dictionary."""
            return {"inputs": self.inputs[idx], "targets": self.targets[idx]}

    dataset = DictDataset(inputs, targets)
    sampler: Any = DistributedSampler(dataset)

    loader = DataLoader(dataset, batch_size=2, sampler=sampler)
    return {"loader": loader, "distributed": getattr(config, "distributed", False)}


@pytest.fixture
def _mock_torch_env(monkeypatch: pytest.MonkeyPatch) -> typing.Iterator[None]:
    """Initialize a single-process gloo distributed group and tiny model stub."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=0, world_size=1)
    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)

    class DummyModel(nn.Module):
        """Tiny embedding module for DDP/FSDP tests."""

        def __init__(self) -> None:
            """Initialize DummyModel."""
            super().__init__()
            self.emb = nn.Embedding(128, 128)

        def forward(self, x: Any, *_args: Any, **_kwargs: Any) -> Any:
            """Compute token embeddings."""
            return self.emb(x.long())

    monkeypatch.setattr(
        "gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM.from_pretrained",
        lambda *_a, **_k: DummyModel(),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    yield
    dist.init_process_group("gloo", rank=0, world_size=1) if not dist.is_initialized() else None
    dist.destroy_process_group()


@pytest.mark.usefixtures("_mock_torch_env")
def test_train_model_ddp() -> None:
    """Verify train_model executes under DDP distributed strategy."""
    res = tr.train_model(
        TrainingConfig(
            action="sft",
            model_name="mod",
            dataset="dat",
            epochs=1,
            learning_rate=0.1,
            distributed_strategy="ddp",
            backend="pytorch_native",
        )
    )
    assert res["status"] == "completed"
    assert res.get("distributed_strategy") == "ddp"


@pytest.mark.usefixtures("_mock_torch_env")
def test_train_model_fsdp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify train_model executes under FSDP distributed strategy."""

    class DummyFSDP(nn.Module):
        """Lightweight FSDP wrapper for CPU testing."""

        def __init__(self, module: Any) -> None:
            """Wrap target module."""
            super().__init__()
            self.module = module

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            """Forward to wrapped module."""
            return self.module(*args, **kwargs)

    class MockFSDPModule:
        """Stub for torch.distributed.fsdp."""

        FullyShardedDataParallel = DummyFSDP

    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", MockFSDPModule)

    res = tr.train_model(
        TrainingConfig(
            action="sft",
            model_name="mod",
            dataset="dat",
            epochs=1,
            learning_rate=0.1,
            distributed_strategy="fsdp",
            backend="pytorch_native",
        )
    )
    assert res["status"] == "completed"
    assert res.get("distributed_strategy") == "fsdp"


def test_export_distributed_rank_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify rank-0 process exports safetensors weights in distributed mode."""
    saved_files: list[str] = []
    monkeypatch.setattr(ex, "save_file", lambda _tensors, filename: saved_files.append(filename))

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("gloo", rank=0, world_size=1)

    try:
        res = ex.export_model("test_model", str(tmp_path), backend_alias="pytorch_native")
        assert res["status"] == "exported_with_safetensors"
        assert len(saved_files) == 1
    finally:
        dist.destroy_process_group()


def test_export_distributed_rank_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify non-rank-0 process skips safetensors export in distributed mode."""
    saved_files: list[str] = []
    monkeypatch.setattr(ex, "save_file", lambda _tensors, filename: saved_files.append(filename))

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    dist.init_process_group("gloo", rank=0, world_size=1)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)

    try:
        res = ex.export_model("test_model", str(tmp_path), backend_alias="pytorch_native")
        assert res["status"] == "skipped_non_rank_zero"
        assert saved_files == []
    finally:
        dist.destroy_process_group()


def test_etl_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify build_dataloader attaches a real DistributedSampler when distributed=True."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"
    dist.init_process_group("gloo", rank=0, world_size=1)

    try:
        monkeypatch.setattr(
            etl,
            "datasets",
            MagicMock(load_dataset=lambda *_a, **_k: [{"question": "SELECT 1", "query": "SELECT 1;"}]),
        )
        monkeypatch.setattr(etl, "SQLTokenizer", SQLTokenizer)

        res = etl.build_dataloader(ETLConfig(dataset_name="ds", split="train", distributed=True))
        assert res.get("distributed") is True
        loader: Any = res.get("loader")
        assert isinstance(loader.sampler, DistributedSampler)
    finally:
        dist.destroy_process_group()
