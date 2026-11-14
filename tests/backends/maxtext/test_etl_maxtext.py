"""Unit and behavioral tests for MaxText Grain ETL pipeline without MagicMock."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pytest

import gemma_4_sql.backends.maxtext as maxtext_pkg
import gemma_4_sql.backends.maxtext.etl as maxtext_etl
from gemma_4_sql.backends.common_data import TokenizedElementDict
from gemma_4_sql.backends.maxtext.etl import build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig, JSONDict, JSONValue


class _DummyBase:
    """Base class stub for Grain `RandomAccessDataSource` and `MapTransform`."""


class _MapOperationProtocol(Protocol):
    """Protocol for a Grain `MapTransform` operation."""

    def map(self, element: dict[str, JSONValue]) -> TokenizedElementDict:
        """Map a dataset element to tokenized inputs, targets, segment_ids, and positions.

        Args:
            element: Input dataset row dictionary.

        Returns:
            Tokenized causal LM dictionary with MaxText `segment_ids` and `positions`.

        """
        ...


class _FakeDatasetsModule:
    """Typed substitute for `datasets` satisfying `HFDatasetsModuleProtocol`."""

    def load_dataset(self, path: str, *, split: str = "train") -> list[JSONDict]:
        """Return a single synthetic Text-to-SQL row.

        Args:
            path: Dataset identifier.
            split: Dataset split.

        Returns:
            List containing one synthetic SQL prompt/target dictionary.

        """
        _ = (path, split)
        return [{"sql_prompt": "SELECT 1", "sql": "SELECT 1;"}]


class _FakeIndexSampler:
    """Typed substitute for `grain.IndexSampler`."""

    def __init__(
        self,
        *,
        num_records: int,
        shard_options: JSONValue = None,
        shuffle: bool = False,
        num_epochs: int | None = 1,
        seed: int = 0,
    ) -> None:
        """Store sampler parameters."""
        self.num_records = num_records
        self.shard_options = shard_options
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.seed = seed


class _FakeShardByJaxProcess:
    """Typed substitute for `grain.ShardByJaxProcess`."""

    def __init__(self, *, drop_remainder: bool = True) -> None:
        """Store `drop_remainder` flag."""
        self.drop_remainder = drop_remainder


class _FakeNoSharding:
    """Typed substitute for `grain.NoSharding`."""


class _FakeBatch:
    """Typed substitute for `grain.Batch`."""

    def __init__(self, *, batch_size: int, drop_remainder: bool = False) -> None:
        """Store batch parameters."""
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder


def test_maxtext_etl_build_dataloader_real_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `maxtext.etl.build_dataloader` wraps `BaseFormatTransform` with MaxText `segment_ids` and `positions`."""
    assert maxtext_pkg.get_trainer() == "maxtext_trainer"
    assert _FakeShardByJaxProcess(drop_remainder=False).drop_remainder is False

    class FakeGrainDataLoader:
        """Lightweight Grain DataLoader substitute executing real `MaxTextFormatTransform`."""

        def __init__(
            self,
            *,
            data_source: Sequence[dict[str, JSONValue]],
            sampler: _FakeIndexSampler,
            operations: Sequence[_MapOperationProtocol | _FakeBatch],
            worker_count: int = 0,
        ) -> None:
            """Store arguments and transform the first record."""
            self.data_source = data_source
            self.sampler = sampler
            first_op = operations[0]
            assert not isinstance(first_op, _FakeBatch)
            self.first_mapped: TokenizedElementDict = first_op.map(data_source[0])
            self.worker_count = worker_count

    class FakeGrainModule:
        """Typed `GrainModuleProtocol` substitute."""

        RandomAccessDataSource = _DummyBase
        MapTransform = _DummyBase
        DataLoader = FakeGrainDataLoader
        IndexSampler = _FakeIndexSampler
        ShardByJaxProcess = _FakeShardByJaxProcess
        NoSharding = _FakeNoSharding
        Batch = _FakeBatch

    monkeypatch.setattr(maxtext_etl, "datasets", _FakeDatasetsModule())
    monkeypatch.setattr(maxtext_etl, "grain", FakeGrainModule())

    config = ETLConfig(dataset_name="ds", split="split", batch_size=2, distributed=True, tokenizer_name="tok")
    res = build_dataloader(config)
    assert res["dataset"] == "ds"
    assert res["split"] == "split"
    assert res["status"] == "loaded"
    assert res["batch_size"] == 2
    assert res["backend"] == "maxtext"
    assert res.get("distributed") is True

    loader = res.get("loader")
    assert isinstance(loader, FakeGrainDataLoader)
    mapped = loader.first_mapped
    assert "inputs" in mapped
    assert "targets" in mapped
    inputs_seq = mapped["inputs"]
    assert isinstance(inputs_seq, list)
    assert mapped["segment_ids"] == [1] * len(inputs_seq)
    assert mapped["positions"] == list(range(len(inputs_seq)))


def test_build_dataloader_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `build_dataloader` raises `DependencyMissingError` when `datasets` is `None`."""
    config = ETLConfig(dataset_name="ds", split="split")
    monkeypatch.setattr(maxtext_etl, "datasets", None)
    with pytest.raises(DependencyMissingError, match=r"Missing grain or datasets\. Cannot load ds\."):
        build_dataloader(config)
