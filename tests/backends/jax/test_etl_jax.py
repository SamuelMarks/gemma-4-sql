"""Module docstring."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Docstring for mock_dependencies."""
    mock_datasets = MagicMock()
    mock_grain = MagicMock()

    # To fix grain.python issues
    mock_grain.python = mock_grain

    mock_datasets.load_dataset.return_value = "mock_hf_dataset"

    mock_grain.JAXDistributedSharding.return_value = "dist_shard"
    mock_grain.NoSharding.return_value = "no_shard"
    mock_grain.IndexSampler.return_value = "mock_sampler"
    mock_grain.DataLoader.return_value = "mock_dataloader"
    mock_grain.Batch.return_value = "mock_batch_op"

    with patch.dict(
        sys.modules,
        {
            "datasets": mock_datasets,
            "grain": mock_grain,
            "grain.python": mock_grain,
        },
    ):
        yield mock_datasets, mock_grain


def reload_module():
    """Docstring for reload_module."""
    import gemma_4_sql.backends.jax.etl as jax_etl

    importlib.reload(jax_etl)
    return jax_etl


def test_missing_dependencies():
    """Docstring for test_missing_dependencies."""
    with patch.dict(sys.modules, {"datasets": None, "grain.python": None, "grain": None}):
        jax_etl = reload_module()

        with pytest.raises(Exception, match="Datasets dependency is missing."):
            jax_etl._load_hf_or_duckdb("ds", "split", None, None)

        with pytest.raises(Exception, match="Grain dependency is missing."):
            jax_etl._get_sampler(10, False)

        with pytest.raises(Exception, match="Missing grain or datasets. Cannot load"):
            config = MagicMock()
            config.dataset_name = "test"
            config.split = "train"
            config.batch_size = 1
            config.distributed = False
            config.tokenizer_name = "t"
            config.duckdb_path = None
            config.duckdb_table = None
            jax_etl.build_dataloader(config)


def test_load_hf_or_duckdb():
    """Docstring for test_load_hf_or_duckdb."""
    jax_etl = reload_module()

    with patch("gemma_4_sql.backends.jax.etl.load_duckdb_dataset") as mock_duckdb:
        mock_duckdb.return_value = "mock_duckdb_ds"

        res = jax_etl._load_hf_or_duckdb("ds", "split", "path", "table")
        assert res == "mock_duckdb_ds"

        res = jax_etl._load_hf_or_duckdb("ds", "split", None, None)
        assert res == "mock_hf_dataset"


def test_get_sampler():
    """Docstring for test_get_sampler."""
    jax_etl = reload_module()

    sampler = jax_etl._get_sampler(10, True)
    assert sampler == "mock_sampler"
    jax_etl.grain.JAXDistributedSharding.assert_called_once()

    jax_etl.grain.IndexSampler.reset_mock()
    sampler = jax_etl._get_sampler(10, False)
    assert sampler == "mock_sampler"
    jax_etl.grain.NoSharding.assert_called_once()


def test_get_sampler_missing_sharding():
    """Docstring for test_get_sampler_missing_sharding."""
    jax_etl = reload_module()

    del jax_etl.grain.JAXDistributedSharding
    del jax_etl.grain.NoSharding

    sampler = jax_etl._get_sampler(10, True)
    assert sampler == "mock_sampler"


def test_build_dataloader():
    """Docstring for test_build_dataloader."""
    jax_etl = reload_module()

    config = MagicMock()
    config.dataset_name = "ds"
    config.split = "train"
    config.batch_size = 2
    config.distributed = True
    config.tokenizer_name = "tok"
    config.duckdb_path = None
    config.duckdb_table = None

    mock_source = MagicMock()
    mock_source.__len__.return_value = 100

    with patch("gemma_4_sql.backends.jax.etl.get_grain_classes", return_value=(MagicMock(return_value=mock_source), MagicMock(return_value="mock_transform"))), patch("gemma_4_sql.backends.jax.etl.SQLTokenizer", return_value="mock_tokenizer"):
        res = jax_etl.build_dataloader(config, duckdb_path="override")
        assert res["dataset"] == "ds"
        assert res["split"] == "train"
        assert res["status"] == "loaded"
        assert res["batch_size"] == 2
        assert res["backend"] == "jax"
        assert res["distributed"]
        assert res["loader"] == "mock_dataloader"
