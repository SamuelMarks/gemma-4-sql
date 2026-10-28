"""Unit and behavioral tests for JAX Grain ETL pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
from unittest.mock import MagicMock

from tests.conftest import requires_jax

pytestmark = requires_jax

import duckdb
import pytest

import gemma_4_sql.backends.jax.etl as jax_etl
from gemma_4_sql.backends.common_data import TokenizedElementDict
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig, JSONValue


class _DummyBase:
    """Base class stub for Grain RandomAccessDataSource and MapTransform."""


class _MapOperationProtocol(Protocol):
    """Protocol for a Grain MapTransform operation."""

    def map(self, element: dict[str, JSONValue]) -> TokenizedElementDict:
        """Map a dataset element to tokenized inputs and targets."""
        ...


def test_jax_etl_build_dataloader_real_transform(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    """Verify jax.etl.build_dataloader loads records from DuckDB and tokenizes them with real SQLTokenizer."""
    in_memory_duckdb.execute("CREATE TABLE sql_train (sql_prompt VARCHAR, sql VARCHAR);")
    in_memory_duckdb.execute("INSERT INTO sql_train VALUES ('Count users', 'SELECT COUNT(*) FROM users;');")

    captured_worker_counts: list[int] = []

    class FakeGrainDataLoader:
        """Lightweight Grain DataLoader substitute that executes the real operations pipeline."""

        def __init__(
            self,
            *,
            data_source: Sequence[dict[str, JSONValue]],
            sampler: JSONValue,
            operations: Sequence[_MapOperationProtocol],
            worker_count: int = 0,
        ) -> None:
            """Record DataLoader arguments and run the map operation on element 0."""
            self.data_source = data_source
            self.sampler = sampler
            self.operations = operations
            captured_worker_counts.append(worker_count)
            self.first_mapped: TokenizedElementDict = operations[0].map(data_source[0])

    mock_grain = MagicMock()
    mock_grain.RandomAccessDataSource = _DummyBase
    mock_grain.MapTransform = _DummyBase
    mock_grain.DataLoader = FakeGrainDataLoader
    monkeypatch.setattr(jax_etl, "grain", mock_grain)
    monkeypatch.setattr(jax_etl, "datasets", MagicMock())
    monkeypatch.setattr(
        "gemma_4_sql.backends.common_data.load_duckdb_dataset",
        lambda _p, _t: [{"sql_prompt": "Count users", "sql": "SELECT COUNT(*) FROM users;"}],  # pyright: ignore[reportUnknownLambdaType]
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
    assert "inputs" in loader.first_mapped
    assert "targets" in loader.first_mapped
    inputs_seq = loader.first_mapped["inputs"]
    assert isinstance(inputs_seq, list) and len(inputs_seq) > 0
    assert captured_worker_counts == [0]


def test_jax_etl_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify jax.etl.build_dataloader raises DependencyMissingError when grain or datasets is missing."""
    monkeypatch.setattr(jax_etl, "datasets", None)
    monkeypatch.setattr(jax_etl, "grain", None)
    cfg = ETLConfig(dataset_name="test", split="train")
    with pytest.raises(DependencyMissingError, match=r"Missing grain or datasets\. Cannot load"):
        jax_etl.build_dataloader(cfg)
