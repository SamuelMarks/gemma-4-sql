import pytest
from unittest import mock
import sys
import gemma_4_sql.backends.maxtext.etl as etl_maxtext

class MockDatasets:
    @staticmethod
    def load_dataset(name: str, split: str) -> list[dict]:
        return [{"question": "Q1", "query": "A1"}]

class MockGrain:
    pass

def test_maxtext_etl_duckdb_missing():
    original_duckdb = getattr(etl_maxtext, "duckdb", None)
    original_datasets = getattr(etl_maxtext, "datasets", None)
    original_grain = getattr(etl_maxtext, "grain", None)
    try:
        etl_maxtext.duckdb = None
        etl_maxtext.datasets = MockDatasets()
        etl_maxtext.grain = MockGrain()
        with pytest.raises(ImportError, match="duckdb is required for DuckDB support"):
            etl_maxtext.build_dataloader("dummy", "train", 10, duckdb_path=":memory:", duckdb_table="tbl")
    finally:
        etl_maxtext.duckdb = original_duckdb
        etl_maxtext.datasets = original_datasets
        etl_maxtext.grain = original_grain
