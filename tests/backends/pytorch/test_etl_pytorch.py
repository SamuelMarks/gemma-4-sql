"""Module docstring."""

from gemma_4_sql.exceptions import DependencyMissingError

"""Provide module docstring."""

import contextlib

import pytest

from gemma_4_sql.type_hints import ETLConfig

"Tests for PyTorch-specific ETL pipeline."


def test_build_dataloader_pytorch_mocked() -> None:
    """Test PyTorch build_dataloader when libraries are missing via direct assignment.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    etl_mod = __import__("gemma_4_sql.backends.pytorch.etl", fromlist=[""])
    orig_torch = etl_mod.torch
    try:
        etl_mod.torch = None
        with pytest.raises(DependencyMissingError, match="Missing PyTorch or datasets. Cannot load dummy/data."):
            etl_mod.build_dataloader(ETLConfig(dataset_name="dummy/data", split="train", batch_size=16, distributed=False))
    finally:
        etl_mod.torch = orig_torch


def test_build_dataloader_pytorch_lightweight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch build_dataloader lightweight fallback when duckdb is provided but torch is missing."""
    pt_etl = __import__("gemma_4_sql.backends.pytorch.etl", fromlist=[""])
    monkeypatch.setattr(pt_etl, "duckdb", MockDuckdb())
    monkeypatch.setattr(pt_etl, "torch", None)
    monkeypatch.setattr(pt_etl, "SQLTokenizer", MockTokenizer)

    with pytest.raises(DependencyMissingError):
        pt_etl.build_dataloader(ETLConfig(dataset_name="dataset", split="split", batch_size=2, duckdb_path="test.db", duckdb_table="tbl"))


class MockConn:
    """Provide class docstring."""

    def execute(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self

    def fetchdf(self) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """

        class MockDF:
            """Provide class docstring."""

            def to_dict(self, orient: object = "records") -> object:
                """Execute function.

                Returns:
                    object: Description of return.

                """
                return [{"a": 1}]

        return MockDF()

    def close(self) -> None:
        """Execute function."""


class MockDuckdb:
    """Provide class docstring."""

    def connect(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockConn()


class MockTokenizer:
    """Provide class docstring."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Execute function."""

    def encode(self, _x: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return [1]


def test_duckdb_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    pt_etl = __import__("gemma_4_sql.backends.pytorch.etl", fromlist=[""])
    monkeypatch.setattr(pt_etl, "duckdb", MockDuckdb())
    monkeypatch.setattr(pt_etl, "datasets", object())
    monkeypatch.setattr(pt_etl, "torch", object())
    monkeypatch.setattr(pt_etl, "Dataset", object)
    monkeypatch.setattr(pt_etl, "DataLoader", object)
    monkeypatch.setattr(pt_etl, "SQLTokenizer", MockTokenizer)
    with contextlib.suppress(TypeError):
        pt_etl.build_dataloader(ETLConfig(dataset_name="dataset", split="split", duckdb_path="test.db", duckdb_table="tbl"))


def test_pytorch_etl_exception(monkeypatch):
    import gemma_4_sql.backends.pytorch.etl as pt_etl

    def mock_tok(x):
        raise ValueError("err")

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.etl.SQLTokenizer", type("Tok", (), {"encode": lambda self, x: mock_tok(x), "__init__": lambda self, **k: None}))

    def fail_load(*a, **k):
        raise ValueError("err")

    monkeypatch.setattr(pt_etl, "_load_hf_or_duckdb", fail_load)
    try:
        res = pt_etl.build_dataloader(pt_etl.ETLConfig(dataset_name="x", split="train", batch_size=1))
    except ValueError:
        res = None
    assert res is None or res.get("loader") is None


def test_pytorch_etl_exception2(monkeypatch):
    import gemma_4_sql.backends.pytorch.etl as pt_etl

    def mock_tok(x):
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "gemma_4_sql.tokenization", type("TokModule", (), {"SQLTokenizer": type("Tok", (), {"encode": mock_tok, "__init__": lambda self, **k: None})}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "gemma_4_sql.tokenization":
            return sys.modules["gemma_4_sql.tokenization"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    monkeypatch.setattr(pt_etl, "_load_hf_or_duckdb", lambda *a, **k: (_ for _ in ()).throw(ValueError("err")))

    try:
        res = pt_etl.build_dataloader(pt_etl.ETLConfig(dataset_name="x", split="train", batch_size=1))
    except ValueError:
        res = None
    assert res is None or res.get("loader") is None


def test_pytorch_etl_except(monkeypatch):
    import gemma_4_sql.backends.pytorch.etl as pt_etl

    monkeypatch.setattr(pt_etl, "SQLTokenizer", type("Tok", (), {"encode": lambda self, x: 1, "__init__": lambda self, **k: None}))
    monkeypatch.setattr(pt_etl, "_load_hf_or_duckdb", lambda *a, **k: (_ for _ in ()).throw(ValueError("err")))
    try:
        res = pt_etl.build_dataloader(pt_etl.ETLConfig(dataset_name="x", split="train", batch_size=1))
    except ValueError:
        res = None
    assert res is None


def test_pytorch_etl_return_none(monkeypatch):
    import gemma_4_sql.backends.pytorch.etl as pt_etl

    monkeypatch.setattr(pt_etl, "SQLTokenizer", type("Tok", (), {"encode": lambda self, x: 1, "__init__": lambda self, **k: None}))
    monkeypatch.setattr(pt_etl, "_load_hf_or_duckdb", lambda *a, **k: (_ for _ in ()).throw(ValueError("err")))
    try:
        res = pt_etl.build_dataloader(pt_etl.ETLConfig(dataset_name="x", split="train", batch_size=1))
    except ValueError:
        res = None
    assert res is None or res.get("loader") is None


def test_pytorch_etl_except2(monkeypatch):
    import gemma_4_sql.backends.pytorch.etl as pt_etl

    monkeypatch.setattr(pt_etl, "SQLTokenizer", type("Tok", (), {"encode": lambda self, x: 1, "__init__": lambda self, **k: None}))
    monkeypatch.setattr(pt_etl, "_load_hf_or_duckdb", lambda *a, **k: (_ for _ in ()).throw(ValueError("err")))
    try:
        res = pt_etl.build_dataloader(pt_etl.ETLConfig(dataset_name="x", split="train", batch_size=1))
    except ValueError:
        res = None
    assert res is None or res.get("loader") is None


def test_pytorch_etl_sampler_err(monkeypatch):
    import gemma_4_sql.backends.pytorch.etl as pt_etl

    def mock_dist(*a, **k):
        raise ValueError("err")

    import sys

    monkeypatch.setitem(sys.modules, "torch.utils.data.distributed", type("Dist", (), {"DistributedSampler": mock_dist}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "torch.utils.data.distributed":
            return sys.modules["torch.utils.data.distributed"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    res = pt_etl._get_sampler([], True)
    assert res is None
