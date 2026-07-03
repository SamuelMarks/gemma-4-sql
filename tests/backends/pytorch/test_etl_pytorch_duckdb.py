"""Provide module docstring."""

import pytest

import gemma_4_sql.backends.pytorch.etl as etl_pytorch


class MockDatasets:
    """Initialize class MockDatasets."""

    @staticmethod
    def load_dataset(*_args: object, **_kwargs: object) -> list[dict]:
        """Initialize function load_dataset.

        Args:
        ----
        name: Description of name.
        split: Description of split.

        """
        return [{"question": "Q1", "query": "A1"}]


class MockDataLoader:
    """Initialize class MockDataLoader."""


class MockDataset:
    """Initialize class MockDataset."""


class MockTorch:
    """Initialize class MockTorch."""


def test_pytorch_etl_duckdb_missing() -> object:
    """Initialize function test_pytorch_etl_duckdb_missing."""
    original_duckdb = getattr(etl_pytorch, "duckdb", None)
    original_datasets = getattr(etl_pytorch, "datasets", None)
    original_dl = getattr(etl_pytorch, "DataLoader", None)
    original_ds = getattr(etl_pytorch, "Dataset", None)
    original_torch = getattr(etl_pytorch, "torch", None)
    try:
        etl_pytorch.duckdb = None
        etl_pytorch.datasets = MockDatasets()
        etl_pytorch.DataLoader = MockDataLoader()
        etl_pytorch.Dataset = MockDataset()
        etl_pytorch.torch = MockTorch()
        with pytest.raises(Exception, match=r".*"):
            etl_pytorch.build_dataloader("dummy", "train", 10, duckdb_path=":memory:", duckdb_table="tbl")
    finally:
        etl_pytorch.duckdb = original_duckdb
        etl_pytorch.datasets = original_datasets
        etl_pytorch.DataLoader = original_dl
        etl_pytorch.Dataset = original_ds
        etl_pytorch.torch = original_torch
