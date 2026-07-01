"""Module docstring."""

import pytest

import gemma_4_sql.backends.keras.etl as etl_keras


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


def test_keras_etl_duckdb_missing() -> object:  # type: ignore[return]
    """Initialize function test_keras_etl_duckdb_missing."""
    original_duckdb = getattr(etl_keras, "duckdb", None)
    original_datasets = getattr(etl_keras, "datasets", None)
    original_grain = getattr(etl_keras, "grain", None)
    try:
        etl_keras.duckdb = None  # type: ignore[attr-defined]
        etl_keras.datasets = MockDatasets()  # type: ignore[attr-defined]
        etl_keras.grain = MockGrain()  # type: ignore[attr-defined]
        with pytest.raises(Exception):  # noqa: B017
            etl_keras.build_dataloader("dummy", "train", 10, duckdb_path=":memory:", duckdb_table="tbl")
    finally:
        etl_keras.duckdb = original_duckdb  # type: ignore[attr-defined]
        etl_keras.datasets = original_datasets  # type: ignore[attr-defined]
        etl_keras.grain = original_grain  # type: ignore[attr-defined]


def test_keras_etl_duckdb_real(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockDuckDBConn:
        def execute(self, *args: object, **kwargs: object) -> object:
            class MockResult:
                def fetchdf(self) -> object:
                    class MockDF:
                        def to_dict(self, orient: str) -> object:
                            return [{"question": "Q1", "query": "A1"}]

                    return MockDF()

            return MockResult()

        def close(self) -> None:
            pass

    class MockDuckDB:
        def connect(self, *args: object, **kwargs: object) -> object:
            return MockDuckDBConn()

    monkeypatch.setattr(etl_keras, "duckdb", MockDuckDB())
    monkeypatch.setattr(etl_keras, "datasets", MockDatasets())
    monkeypatch.setattr(
        etl_keras, "grain", type("MockGrain", (), {"JAXDistributedSharding": lambda: None, "NoSharding": lambda: None, "IndexSampler": lambda *args, **kwargs: None, "DataLoader": lambda *args, **kwargs: "loader", "RandomAccessDataSource": object, "MapTransform": object, "Batch": lambda **kwargs: "batch"})
    )
    res = etl_keras.build_dataloader("ds", "train", duckdb_path=":memory:", duckdb_table="t")
    assert res["status"] == "loaded"
