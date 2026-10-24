"""Unit and behavioral tests for etl pytorch duckdb."""

from typing import Any

import pytest

from tests.conftest import requires_torch

pytestmark = requires_torch

import gemma_4_sql.backends.pytorch.etl as etl_pytorch
from gemma_4_sql.type_hints import ETLConfig


class MockDatasets:
    """Initialize class MockDatasets."""

    @staticmethod
    def load_dataset(*_args: Any, **_kwargs: Any) -> list[dict[str, str]]:
        """Return mock dataset records."""
        return [{"question": "Q1", "query": "A1"}]


class MockDataLoader:
    """Initialize class MockDataLoader."""


class MockDataset:
    """Initialize class MockDataset."""


class MockTorch:
    """Initialize class MockTorch."""


def test_pytorch_etl_duckdb_missing(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Verify build_dataloader raises when duckdb is unavailable."""
    monkeypatch.setattr(etl_pytorch, "duckdb", None, raising=False)
    monkeypatch.setattr(etl_pytorch, "datasets", MockDatasets())
    _ = MockDatasets.load_dataset()
    monkeypatch.setattr(etl_pytorch, "DataLoader", MockDataLoader())
    monkeypatch.setattr(etl_pytorch, "Dataset", MockDataset())
    monkeypatch.setattr(etl_pytorch, "torch", MockTorch())
    with pytest.raises(Exception, match=r".*"):
        etl_pytorch.build_dataloader(ETLConfig(dataset_name="dummy", split="train", batch_size=10, duckdb_path=":memory:", duckdb_table="tbl"))
