import importlib
import sys

import pytest


def test_duckdb_conn(in_memory_duckdb):
    # just getting the fixture covers it
    assert in_memory_duckdb is not None


@pytest.mark.asyncio
async def test_sqlite_conn(in_memory_sqlite) -> None:
    assert in_memory_sqlite is not None


def test_mock_dataset(synthetic_dataset):
    assert len(synthetic_dataset) == 2


def test_missing_dependency_context(missing_dependency_context):
    with missing_dependency_context(["fake_module_xyz"]):
        assert sys.modules.get("fake_module_xyz") is None


def test_torch_import_fallback(monkeypatch):
    import builtins

    from tests import conftest

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("mock missing torch")
        return orig_import(name, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mock_import)
        importlib.reload(conftest)
        assert conftest.torch is None

    # reload it back
    importlib.reload(conftest)
