"""Provide module docstring."""

import pytest

import gemma_4_sql.backends.keras.etl as etl_keras


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


class MockGrain:
    """Initialize class MockGrain."""


def test_keras_etl_duckdb_missing() -> object:
    """Initialize function test_keras_etl_duckdb_missing."""
    original_duckdb = getattr(etl_keras, "duckdb", None)
    original_datasets = getattr(etl_keras, "datasets", None)
    original_grain = getattr(etl_keras, "grain", None)
    try:
        etl_keras.duckdb = None
        etl_keras.datasets = MockDatasets()
        etl_keras.grain = MockGrain()
        with pytest.raises(Exception, match=r".*"):
            etl_keras.build_dataloader("dummy", "train", 10, duckdb_path=":memory:", duckdb_table="tbl")
    finally:
        etl_keras.duckdb = original_duckdb
        etl_keras.datasets = original_datasets
        etl_keras.grain = original_grain


class MockDuckDBConn:
    """Provide class docstring."""

    def execute(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""

        class MockResult:
            """Provide class docstring."""

            def fetchdf(self) -> object:
                """Execute function."""

                class MockDF:
                    """Provide class docstring."""

                    def to_dict(self, _orient: str) -> object:
                        """Execute function."""
                        return [{"question": "Q1", "query": "A1"}]

                return MockDF()

        return MockResult()

    def close(self) -> None:
        """Execute function."""


class MockDuckDB:
    """Provide class docstring."""

    def connect(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return MockDuckDBConn()


def test_keras_etl_duckdb_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(etl_keras, "duckdb", MockDuckDB())
    monkeypatch.setattr(etl_keras, "datasets", MockDatasets())
    monkeypatch.setattr(
        etl_keras,
        "grain",
        type(
            "MockGrain",
            (),
            {"JAXDistributedSharding": lambda: None, "NoSharding": lambda: None, "IndexSampler": lambda *_args, **_kwargs: None, "DataLoader": lambda *_args, **_kwargs: "loader", "RandomAccessDataSource": type("Dummy", (), {}), "MapTransform": type("Dummy", (), {}), "Batch": lambda **_kwargs: "batch"},
        ),
    )
    res = etl_keras.build_dataloader("ds", "train", duckdb_path=":memory:", duckdb_table="t")
    if res["status"] != "loaded":
        raise AssertionError
