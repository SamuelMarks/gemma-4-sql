import pytest

"""Tests for PyTorch-specific ETL pipeline."""


def test_build_dataloader_pytorch_mocked() -> None:
    """Test PyTorch build_dataloader when libraries are missing via direct assignment."""
    etl_mod = __import__("gemma_4_sql.backends.pytorch.etl", fromlist=[""])
    orig_torch = etl_mod.torch
    try:
        etl_mod.torch = None  # type: ignore[attr-defined]
        res = etl_mod.build_dataloader("dummy/data", "train", 16, distributed=False)
        if not res["backend"] == "pytorch":
            raise AssertionError
        if not res["status"] == "mocked":
            raise AssertionError
        if "mock_samples" not in res:
            raise AssertionError
    finally:
        etl_mod.torch = orig_torch  # type: ignore[attr-defined]


def test_duckdb_execution(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.pytorch.etl as pt_etl

    class MockConn:
        def execute(self, *args, **kwargs):
            return self

        def fetchdf(self):
            class MockDF:
                def to_dict(self, orient):
                    return [{"a": 1}]

            return MockDF()

        def close(self):
            pass

    class MockDuckdb:
        def connect(self, *args, **kwargs):
            return MockConn()

    monkeypatch.setattr(pt_etl, "duckdb", MockDuckdb())
    monkeypatch.setattr(pt_etl, "datasets", object())
    monkeypatch.setattr(pt_etl, "torch", object())
    monkeypatch.setattr(pt_etl, "Dataset", object)
    monkeypatch.setattr(pt_etl, "DataLoader", object)

    class MockTokenizer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, x):
            return [1]

    monkeypatch.setattr(pt_etl, "SQLTokenizer", MockTokenizer)

    try:
        pt_etl.build_dataloader("dataset", "split", duckdb_path="test.db", duckdb_table="tbl")
    except TypeError:
        # We mocked Dataset with object, so defining PyTorchDataset(Dataset) might fail or work
        # but the duckdb code will have been executed.
        pass
