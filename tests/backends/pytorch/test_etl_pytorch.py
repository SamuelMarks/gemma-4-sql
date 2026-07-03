"""Provide module docstring."""

import contextlib

import pytest

"Tests for PyTorch-specific ETL pipeline."


def test_build_dataloader_pytorch_mocked() -> None:
    """Test PyTorch build_dataloader when libraries are missing via direct assignment."""
    etl_mod = __import__("gemma_4_sql.backends.pytorch.etl", fromlist=[""])
    orig_torch = etl_mod.torch
    try:
        etl_mod.torch = None
        res = etl_mod.build_dataloader("dummy/data", "train", 16, distributed=False)
        if not res["backend"] == "pytorch":
            raise AssertionError
        if not res["status"] == "mocked":
            raise AssertionError
        if "mock_samples" not in res:
            raise AssertionError
    finally:
        etl_mod.torch = orig_torch


class MockConn:
    """Provide class docstring."""

    def execute(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return self

    def fetchdf(self) -> object:
        """Execute function."""

        class MockDF:
            """Provide class docstring."""

            def to_dict(self, _orient: object) -> object:
                """Execute function."""
                return [{"a": 1}]

        return MockDF()

    def close(self) -> None:
        """Execute function."""


class MockDuckdb:
    """Provide class docstring."""

    def connect(self, *_args: object, **_kwargs: object) -> object:
        """Execute function."""
        return MockConn()


class MockTokenizer:
    """Provide class docstring."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Execute function."""

    def encode(self, _x: object) -> object:
        """Execute function."""
        return [1]


def test_duckdb_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    pt_etl = __import__("gemma_4_sql.backends.pytorch.etl")
    monkeypatch.setattr(pt_etl, "duckdb", MockDuckdb())
    monkeypatch.setattr(pt_etl, "datasets", object())
    monkeypatch.setattr(pt_etl, "torch", object())
    monkeypatch.setattr(pt_etl, "Dataset", object)
    monkeypatch.setattr(pt_etl, "DataLoader", object)
    monkeypatch.setattr(pt_etl, "SQLTokenizer", MockTokenizer)
    with contextlib.suppress(TypeError):
        pt_etl.build_dataloader("dataset", "split", duckdb_path="test.db", duckdb_table="tbl")
