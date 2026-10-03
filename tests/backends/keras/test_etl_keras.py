"""Tests for Keras ETL."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.keras.etl import _get_sampler, _load_hf_or_duckdb, build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig


def test_load_hf_or_duckdb_duckdb():
    with patch("gemma_4_sql.backends.keras.etl.load_duckdb_dataset", return_value="duckdb") as mock_duckdb:
        res = _load_hf_or_duckdb("ds", "train", "path", "table")
        assert res == "duckdb"
        mock_duckdb.assert_called_once_with("path", "table")


def test_load_hf_or_duckdb_missing_datasets():
    with patch("gemma_4_sql.backends.keras.etl.datasets", None), pytest.raises(DependencyMissingError, match="Datasets dependency is missing"):
        _load_hf_or_duckdb("ds", "train", None, None)


def test_load_hf_or_duckdb_hf():
    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = "hf"
    with patch("gemma_4_sql.backends.keras.etl.datasets", mock_datasets):
        res = _load_hf_or_duckdb("ds", "train", None, None)
        assert res == "hf"
        mock_datasets.load_dataset.assert_called_once_with("ds", split="train")


def test_get_sampler_missing_grain():
    with patch("gemma_4_sql.backends.keras.etl.grain", None), pytest.raises(DependencyMissingError, match="Grain dependency is missing"):
        _get_sampler(10, False)


def test_get_sampler_success():
    mock_grain = MagicMock()
    mock_grain.JAXDistributedSharding.return_value = "dist"
    mock_grain.NoSharding.return_value = "no"
    mock_grain.IndexSampler.return_value = "sampler"
    with patch("gemma_4_sql.backends.keras.etl.grain", mock_grain):
        res1 = _get_sampler(10, True)
        assert res1 == "sampler"
        mock_grain.IndexSampler.assert_called_with(num_records=10, shard_options="dist", shuffle=False, num_epochs=1)

        res2 = _get_sampler(10, False)
        assert res2 == "sampler"
        mock_grain.IndexSampler.assert_called_with(num_records=10, shard_options="no", shuffle=False, num_epochs=1)


def test_build_dataloader_missing_deps():
    with patch("gemma_4_sql.backends.keras.etl.datasets", None), pytest.raises(DependencyMissingError, match="Missing grain or datasets"):
        build_dataloader(ETLConfig(dataset_name="a", split="b", batch_size=2))


def test_build_dataloader_success():
    mock_datasets = MagicMock()
    mock_grain = MagicMock()
    mock_grain.DataLoader.return_value = "loader"
    mock_grain.Batch = MagicMock()

    with (
        patch("gemma_4_sql.backends.keras.etl.datasets", mock_datasets),
        patch("gemma_4_sql.backends.keras.etl.grain", mock_grain),
        patch("gemma_4_sql.backends.keras.etl._load_hf_or_duckdb", return_value=[1, 2]),
        patch("gemma_4_sql.backends.keras.etl.get_grain_classes", return_value=(list, MagicMock)),
        patch("gemma_4_sql.backends.keras.etl.SQLTokenizer"),
        patch("gemma_4_sql.backends.keras.etl._get_sampler"),
    ):
        config = ETLConfig(dataset_name="ds", split="train", batch_size=2, distributed=False, tokenizer_name="tok")
        res = build_dataloader(config)
        assert res["loader"] == "loader"
        assert res["status"] == "loaded"
        assert res["dataset"] == "ds"
