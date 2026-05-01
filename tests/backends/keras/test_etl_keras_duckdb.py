import pytest

import gemma_4_sql.backends.keras.etl as etl_keras


class MockDatasets:
    @staticmethod
    def load_dataset(name: str, split: str) -> list[dict]:
        return [{"question": "Q1", "query": "A1"}]


class MockGrain:
    pass


def test_keras_etl_duckdb_missing():
    original_duckdb = getattr(etl_keras, "duckdb", None)
    original_datasets = getattr(etl_keras, "datasets", None)
    original_grain = getattr(etl_keras, "grain", None)
    try:
        etl_keras.duckdb = None
        etl_keras.datasets = MockDatasets()
        etl_keras.grain = MockGrain()
        with pytest.raises(ImportError, match="duckdb is required for DuckDB support"):
            etl_keras.build_dataloader(
                "dummy", "train", 10, duckdb_path=":memory:", duckdb_table="tbl"
            )
    finally:
        etl_keras.duckdb = original_duckdb
        etl_keras.datasets = original_datasets
        etl_keras.grain = original_grain
