"""Provide module docstring."""

import pytest

import gemma_4_sql.backends.maxtext.etl as etl_maxtext
from gemma_4_sql.type_hints import ETLConfig


def test_maxtext_etl_duckdb_missing() -> object:
    """Initialize function test_maxtext_etl_duckdb_missing."""
    original_duckdb = getattr(etl_maxtext, "duckdb", None)
    original_datasets = getattr(etl_maxtext, "datasets", None)
    original_grain = getattr(etl_maxtext, "grain", None)
    try:
        etl_maxtext.duckdb = None
        etl_maxtext.datasets = object()
        etl_maxtext.grain = object()
        with pytest.raises(Exception, match=r".*"):
            etl_maxtext.build_dataloader(ETLConfig(dataset_name="dummy", split="train", batch_size=10, duckdb_path=":memory:", duckdb_table="tbl"))
    finally:
        etl_maxtext.duckdb = original_duckdb
        etl_maxtext.datasets = original_datasets
        etl_maxtext.grain = original_grain
