"""Unit and behavioral tests for JAX Grain ETL pipeline."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from tests.conftest import requires_jax

pytestmark = requires_jax

import duckdb
import pytest

import gemma_4_sql.backends.jax.etl as jax_etl
from gemma_4_sql.backends.common_data import (
    BaseFormatTransform,
    GrainBatchOpProtocol,
    GrainSamplerProtocol,
    GrainShardingOptionsProtocol,
    HFDataSource,
    TokenizedElementDict,
    _DefaultBaseTransform,
)
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig, JSONDict


class _DummyBase:
    """Base class stub for Grain RandomAccessDataSource and MapTransform."""


class _FakeShardingOptions(GrainShardingOptionsProtocol):
    """Typed Grain sharding options double."""


class _FakeGrainSampler(GrainSamplerProtocol):
    """Typed Grain `IndexSampler` double."""

    def __init__(self, num_records: int, shard_options: GrainShardingOptionsProtocol | None) -> None:
        """Store `num_records` and `shard_options`."""
        self.num_records = num_records
        self.shard_options = shard_options


class _FakeGrainBatchOp(GrainBatchOpProtocol):
    """Typed Grain `Batch` operation double."""

    def __init__(self, batch_size: int) -> None:
        """Store `batch_size`."""
        self.batch_size = batch_size


class _FakeDatasetsModule:
    """Typed Hugging Face `datasets` module double."""

    def load_dataset(self, _path: str, *, split: str = "train") -> list[JSONDict]:
        """Return single-row synthetic SQL dataset."""
        _ = split
        return [{"sql_prompt": "Count users", "sql": "SELECT COUNT(*) FROM users;"}]


def test_jax_etl_build_dataloader_real_transform(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    """Verify jax.etl.build_dataloader loads records from DuckDB and HF and tokenizes them with real SQLTokenizer."""
    in_memory_duckdb.execute("CREATE TABLE sql_train (sql_prompt VARCHAR, sql VARCHAR);")
    in_memory_duckdb.execute("INSERT INTO sql_train VALUES ('Count users', 'SELECT COUNT(*) FROM users;');")

    captured_worker_counts: list[int] = []

    class FakeGrainDataLoader:
        """Lightweight Grain DataLoader substitute that executes the real operations pipeline."""

        def __init__(
            self,
            *,
            data_source: HFDataSource,
            sampler: GrainSamplerProtocol,
            operations: Sequence[_DefaultBaseTransform | GrainBatchOpProtocol],
            worker_count: int = 0,
        ) -> None:
            """Record DataLoader arguments and run the map operation on element 0."""
            self.data_source = data_source
            self.sampler = sampler
            self.operations = operations
            captured_worker_counts.append(worker_count)
            first_op = operations[0]
            assert isinstance(first_op, BaseFormatTransform)
            first_elem = data_source[0]
            assert isinstance(first_elem, dict)
            self.first_mapped: TokenizedElementDict = first_op.map(first_elem)

        def __iter__(self) -> Iterator[TokenizedElementDict]:
            """Iterate over the mapped first element.

            Returns:
                An iterator yielding the tokenized first element.

            """
            return iter([self.first_mapped])

        def __len__(self) -> int:
            """Return number of items in `self.data_source`."""
            return len(self.data_source)

    class _FakeGrainModule:
        """Typed `GrainModuleProtocol` double delegating `DataLoader` to `FakeGrainDataLoader`."""

        RandomAccessDataSource = _DummyBase
        MapTransform = _DummyBase
        NoSharding = _FakeShardingOptions
        JAXDistributedSharding = _FakeShardingOptions

        def IndexSampler(
            self,
            *,
            num_records: int,
            shard_options: GrainShardingOptionsProtocol | None,
            shuffle: bool = False,
            num_epochs: int = 1,
        ) -> _FakeGrainSampler:
            """Construct `_FakeGrainSampler`.

            Returns:
                A `_FakeGrainSampler` configured with record count and sharding.

            """
            _ = (shuffle, num_epochs)
            return _FakeGrainSampler(num_records, shard_options)

        def Batch(self, *, batch_size: int) -> _FakeGrainBatchOp:
            """Construct `_FakeGrainBatchOp`.

            Returns:
                A `_FakeGrainBatchOp` configured with `batch_size`.

            """
            return _FakeGrainBatchOp(batch_size)

        def DataLoader(
            self,
            *,
            data_source: HFDataSource,
            sampler: GrainSamplerProtocol,
            operations: Sequence[_DefaultBaseTransform | GrainBatchOpProtocol],
        ) -> FakeGrainDataLoader:
            """Construct `FakeGrainDataLoader`.

            Returns:
                A `FakeGrainDataLoader` wrapping the data source and operations.

            """
            return FakeGrainDataLoader(
                data_source=data_source,
                sampler=sampler,
                operations=operations,
            )

    monkeypatch.setattr(jax_etl, "grain", _FakeGrainModule())
    monkeypatch.setattr(jax_etl, "datasets", _FakeDatasetsModule())

    def _load_duckdb_stub(_p: str, _t: str) -> list[JSONDict]:
        """Return 1-row SQL training dataset."""
        return [{"sql_prompt": "Count users", "sql": "SELECT COUNT(*) FROM users;"}]

    monkeypatch.setattr(
        "gemma_4_sql.backends.common_data.load_duckdb_dataset",
        _load_duckdb_stub,
    )

    cfg = ETLConfig(
        dataset_name="custom",
        split="train",
        batch_size=2,
        distributed=True,
        duckdb_path=":memory:",
        duckdb_table="sql_train",
    )
    res = jax_etl.build_dataloader(cfg)
    assert res["status"] == "loaded"
    assert res["backend"] == "jax"
    assert res["batch_size"] == 2
    loader = res.get("loader")
    assert isinstance(loader, FakeGrainDataLoader)
    assert len(loader) == 1
    assert len(list(iter(loader))) == 1
    assert "inputs" in loader.first_mapped
    assert "targets" in loader.first_mapped
    inputs_seq = loader.first_mapped["inputs"]
    assert isinstance(inputs_seq, list) and len(inputs_seq) > 0

    res_hf = jax_etl.build_dataloader(ETLConfig(dataset_name="spider", split="train", batch_size=2, distributed=False))
    assert res_hf["status"] == "loaded"
    assert captured_worker_counts == [0, 0]


def test_jax_etl_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify jax.etl.build_dataloader raises DependencyMissingError when grain or datasets is missing."""
    monkeypatch.setattr(jax_etl, "datasets", None)
    monkeypatch.setattr(jax_etl, "grain", None)
    cfg = ETLConfig(dataset_name="test", split="train")
    with pytest.raises(DependencyMissingError, match=r"Missing grain or datasets\. Cannot load"):
        jax_etl.build_dataloader(cfg)
