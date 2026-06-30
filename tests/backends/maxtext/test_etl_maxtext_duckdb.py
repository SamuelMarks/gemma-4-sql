"""Module docstring."""

import gemma_4_sql.backends.maxtext.etl as etl_maxtext
import pytest


class MockDatasets:
    """Initialize class MockDatasets."""

    @staticmethod
    def load_dataset(*_args: object, **_kwargs: object) -> list[dict]:  # type: ignore[type-arg]
        """Initialize function load_dataset.

        Args:
        ----
        name: Description of name.
        split: Description of split.

        """
        return [{"question": "Q1", "query": "A1"}]


class MockGrain:
    """Initialize class MockGrain."""


def test_maxtext_etl_duckdb_missing() -> object:  # type: ignore[return]
    """Initialize function test_maxtext_etl_duckdb_missing."""
    original_duckdb = getattr(etl_maxtext, "duckdb", None)
    original_datasets = getattr(etl_maxtext, "datasets", None)
    original_grain = getattr(etl_maxtext, "grain", None)
    try:
        etl_maxtext.duckdb = None  # type: ignore[attr-defined]
        etl_maxtext.datasets = MockDatasets()  # type: ignore[attr-defined]
        etl_maxtext.grain = MockGrain()  # type: ignore[attr-defined]
        with pytest.raises(ImportError, match="duckdb is required for DuckDB support"):
            etl_maxtext.build_dataloader("dummy", "train", 10, duckdb_path=":memory:", duckdb_table="tbl")
    finally:
        etl_maxtext.duckdb = original_duckdb  # type: ignore[attr-defined]
        etl_maxtext.datasets = original_datasets  # type: ignore[attr-defined]
        etl_maxtext.grain = original_grain  # type: ignore[attr-defined]
