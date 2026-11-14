"""Unit and behavioral tests for Keras Grain ETL pipeline without untyped mocks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from tests.conftest import requires_keras

pytestmark = requires_keras

import pytest

from gemma_4_sql.backends.common_data import (
    GrainBatchOpProtocol,
    GrainSamplerProtocol,
    GrainShardingOptionsProtocol,
    TokenizedElementDict,
)
from gemma_4_sql.backends.keras import etl
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig, JSONDict, JSONValue


class _DummyBase:
    """Base class stub for Grain `RandomAccessDataSource` and `MapTransform`."""


class _FakeSharding(GrainShardingOptionsProtocol):
    """Typed Grain sharding options double."""


class _FakeSampler(GrainSamplerProtocol):
    """Typed Grain `IndexSampler` double."""

    def __init__(self, *, num_records: int, shard_options: GrainShardingOptionsProtocol | None) -> None:
        """Store `num_records` and `shard_options`."""
        self.num_records = num_records
        self.shard_options = shard_options


class _FakeBatchOp(GrainBatchOpProtocol):
    """Typed Grain `Batch` double."""

    def __init__(self, *, batch_size: int, drop_remainder: bool = False) -> None:
        """Store `batch_size` and `drop_remainder`."""
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder


class _FakeDatasets:
    """Typed Hugging Face `datasets` double."""

    def load_dataset(self, _path: str, *, split: str = "train") -> list[JSONDict]:
        """Return single-row synthetic SQL dataset."""
        _ = split
        return [{"sql_prompt": "List tables", "sql": "SELECT name FROM sqlite_master;"}]


class _MapOperationProtocol(Protocol):
    """Protocol for a Grain `MapTransform` operation."""

    def map(self, element: dict[str, JSONValue]) -> TokenizedElementDict:
        """Map a dataset element to tokenized inputs and targets."""
        ...


def test_keras_etl_build_dataloader_real_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `keras.etl.build_dataloader` constructs a real `HFDataSource` and `BaseFormatTransform`."""

    class FakeGrainDataLoader:
        """Lightweight Grain `DataLoader` substitute executing real format transformation."""

        def __init__(
            self,
            *,
            data_source: Sequence[dict[str, JSONValue]],
            sampler: _FakeSampler,
            operations: Sequence[_MapOperationProtocol | _FakeBatchOp],
            worker_count: int = 0,
        ) -> None:
            """Store arguments and transform the first record."""
            self.data_source = data_source
            self.sampler = sampler
            first_op = operations[0]
            assert not isinstance(first_op, _FakeBatchOp)
            self.first_mapped: TokenizedElementDict = first_op.map(data_source[0])
            self.worker_count = worker_count

    class _FakeGrain:
        """Typed `GrainModuleProtocol` substitute."""

        RandomAccessDataSource = _DummyBase
        MapTransform = _DummyBase
        DataLoader = FakeGrainDataLoader
        NoSharding = _FakeSharding
        ShardByJaxProcess = _FakeSharding
        Batch = _FakeBatchOp

        def IndexSampler(
            self,
            *,
            num_records: int,
            shard_options: GrainShardingOptionsProtocol | None,
            shuffle: bool = False,
            num_epochs: int | None = 1,
            seed: int = 0,
        ) -> _FakeSampler:
            """Construct `_FakeSampler`.

            Returns:
                A `_FakeSampler` configured with `num_records` and `shard_options`.

            """
            _ = (shuffle, num_epochs, seed)
            return _FakeSampler(num_records=num_records, shard_options=shard_options)

    monkeypatch.setattr(etl, "datasets", _FakeDatasets())
    monkeypatch.setattr(etl, "grain", _FakeGrain())

    cfg = ETLConfig(dataset_name="spider", split="train", batch_size=4, distributed=False)
    res = etl.build_dataloader(cfg)
    assert res["status"] == "loaded"
    assert res["backend"] == "keras"
    assert res["batch_size"] == 4
    loader = res.get("loader")
    assert isinstance(loader, FakeGrainDataLoader)
    assert len(loader.data_source) == 1
    inputs_seq = loader.first_mapped["inputs"]
    targets_seq = loader.first_mapped["targets"]
    assert isinstance(inputs_seq, list) and len(inputs_seq) > 0
    assert isinstance(targets_seq, list) and len(targets_seq) > 0


def test_keras_etl_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `keras.etl.build_dataloader` raises `DependencyMissingError` when `datasets` or `grain` is `None`."""
    cfg = ETLConfig(dataset_name="spider", split="train")
    monkeypatch.setattr(etl, "datasets", None)
    with pytest.raises(DependencyMissingError):
        etl.build_dataloader(cfg)

    monkeypatch.setattr(etl, "datasets", _FakeDatasets())
    monkeypatch.setattr(etl, "grain", None)
    with pytest.raises(DependencyMissingError):
        etl.build_dataloader(cfg)
