"""Unit and behavioral tests for etl keras."""

from unittest.mock import MagicMock

from tests.conftest import requires_keras

pytestmark = requires_keras

import pytest


def test_load_hf_or_duckdb(monkeypatch: pytest.MonkeyPatch):
    """Verify load hf or duckdb behavior."""
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


def test_get_sampler(monkeypatch: pytest.MonkeyPatch):
    """Verify get sampler behavior."""
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


def test_build_dataloader(monkeypatch: pytest.MonkeyPatch):
    """Verify build dataloader behavior."""
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
        """Test stub for `MockSource`."""

        def __len__(self):
            """Provide len   test helper."""
            return 10

    monkeypatch.setattr(etl, "get_grain_classes", MagicMock(return_value=(MagicMock(return_value=MockSource()), MagicMock())))
    monkeypatch.setattr(etl, "SQLTokenizer", MagicMock())
    monkeypatch.setattr(etl, "_get_sampler", MagicMock())

    res = etl.build_dataloader(config)
    assert res["status"] == "loaded"


def test_build_dataloader_missing_grain(monkeypatch: pytest.MonkeyPatch):
    """Verify build dataloader missing grain behavior."""
    from gemma_4_sql.backends.keras import etl
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(etl, "datasets", MagicMock())
    monkeypatch.setattr(etl, "grain", None)
    config = MagicMock()
    with pytest.raises(DependencyMissingError):
        etl.build_dataloader(config)
