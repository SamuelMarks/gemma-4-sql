"""Module docstring."""

from unittest.mock import MagicMock

import pytest


def test_load_hf_or_duckdb(monkeypatch):
    """Docstring for test_load_hf_or_duckdb."""
    from gemma_4_sql.backends.keras import etl
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr("gemma_4_sql.backends.keras.etl.load_duckdb_dataset", MagicMock(return_value="duckdb"))
    assert etl._load_hf_or_duckdb("ds", "train", "path", "table") == "duckdb"

    monkeypatch.setattr(etl, "datasets", None)
    with pytest.raises(DependencyMissingError):
        etl._load_hf_or_duckdb("ds", "train", None, None)

    mock_ds = MagicMock()
    mock_ds.load_dataset.return_value = "hf"
    monkeypatch.setattr(etl, "datasets", mock_ds)
    assert etl._load_hf_or_duckdb("ds", "train", None, None) == "hf"


def test_import_error():
    """Docstring for test_import_error."""
    import builtins
    import importlib

    import gemma_4_sql.backends.keras.etl as mod

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "datasets" or name == "grain.python":
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        importlib.reload(mod)
        assert mod.datasets is None
        assert mod.grain is None
    finally:
        builtins.__import__ = orig_import
        importlib.reload(mod)


def test_get_sampler(monkeypatch):
    """Docstring for test_get_sampler."""
    from gemma_4_sql.backends.keras import etl
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(etl, "grain", None)
    with pytest.raises(DependencyMissingError):
        etl._get_sampler(10, False)

    mock_grain = MagicMock()
    mock_grain.JAXDistributedSharding.return_value = "dist"
    mock_grain.NoSharding.return_value = "none"
    mock_grain.IndexSampler.return_value = "sampler"
    monkeypatch.setattr(etl, "grain", mock_grain)

    assert etl._get_sampler(10, True) == "sampler"
    assert etl._get_sampler(10, False) == "sampler"


def test_build_dataloader(monkeypatch):
    """Docstring for test_build_dataloader."""
    from gemma_4_sql.backends.keras import etl
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(etl, "datasets", None)
    config = MagicMock()
    with pytest.raises(DependencyMissingError):
        etl.build_dataloader(config)

    monkeypatch.setattr(etl, "datasets", MagicMock())
    monkeypatch.setattr(etl, "grain", MagicMock())
    monkeypatch.setattr(etl, "_load_hf_or_duckdb", MagicMock())

    class MockSource:
        """Docstring for MockSource."""

        def __len__(self):
            """Docstring for __len__."""
            return 10

    monkeypatch.setattr(etl, "get_grain_classes", MagicMock(return_value=(MagicMock(return_value=MockSource()), MagicMock())))
    monkeypatch.setattr(etl, "SQLTokenizer", MagicMock())
    monkeypatch.setattr(etl, "_get_sampler", MagicMock())

    res = etl.build_dataloader(config)
    assert res["status"] == "loaded"


def test_build_dataloader_missing_grain(monkeypatch):
    """Docstring for test_build_dataloader_missing_grain."""
    from gemma_4_sql.backends.keras import etl
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(etl, "datasets", MagicMock())
    monkeypatch.setattr(etl, "grain", None)
    config = MagicMock()
    with pytest.raises(DependencyMissingError):
        etl.build_dataloader(config)
