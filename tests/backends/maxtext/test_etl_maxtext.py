"""Module docstring."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.maxtext.etl import _get_sampler, _load_hf_or_duckdb, build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig


def test_load_hf_or_duckdb():
    """Docstring for test_load_hf_or_duckdb."""
    with patch("gemma_4_sql.backends.maxtext.etl.load_duckdb_dataset", return_value="duckdb_ds") as mock_duckdb:
        res = _load_hf_or_duckdb("ds", "split", "path", "table")
        assert res == "duckdb_ds"
        mock_duckdb.assert_called_once_with("path", "table")

    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = "hf_ds"
    with patch("gemma_4_sql.backends.maxtext.etl.datasets", mock_datasets):
        res = _load_hf_or_duckdb("ds", "split", None, None)
        assert res == "hf_ds"
        mock_datasets.load_dataset.assert_called_once_with("ds", split="split")


def test_load_hf_missing_deps():
    """Docstring for test_load_hf_missing_deps."""
    with patch("gemma_4_sql.backends.maxtext.etl.datasets", None), pytest.raises(DependencyMissingError, match="Datasets dependency is missing."):
        _load_hf_or_duckdb("ds", "split", None, None)


def test_get_sampler():
    """Docstring for test_get_sampler."""
    mock_grain = MagicMock()
    mock_grain.JAXDistributedSharding = MagicMock(return_value="shard")
    mock_grain.NoSharding = MagicMock(return_value="no_shard")
    mock_grain.IndexSampler.return_value = "sampler"

    with patch("gemma_4_sql.backends.maxtext.etl.grain", mock_grain):
        res = _get_sampler(10, True)
        assert res == "sampler"
        mock_grain.JAXDistributedSharding.assert_called_once()
        mock_grain.IndexSampler.assert_called_once_with(num_records=10, shard_options="shard", shuffle=False, num_epochs=1)

        mock_grain.IndexSampler.reset_mock()
        res = _get_sampler(10, False)
        assert res == "sampler"
        mock_grain.NoSharding.assert_called_once()
        mock_grain.IndexSampler.assert_called_once_with(num_records=10, shard_options="no_shard", shuffle=False, num_epochs=1)


def test_get_sampler_missing_deps():
    """Docstring for test_get_sampler_missing_deps."""
    with patch("gemma_4_sql.backends.maxtext.etl.grain", None), pytest.raises(DependencyMissingError, match="Grain dependency is missing."):
        _get_sampler(10, True)


def test_build_dataloader():
    """Docstring for test_build_dataloader."""
    config = ETLConfig(dataset_name="ds", split="split", batch_size=2, distributed=True, tokenizer_name="tok")

    mock_datasets = MagicMock()
    mock_grain = MagicMock()
    mock_grain.DataLoader.return_value = "loader"
    mock_grain.Batch = MagicMock(return_value="batch_op")

    class MockBaseTransform:
        """Docstring for MockBaseTransform."""

        def __init__(self, tokenizer):
            """Docstring for __init__."""

        def map(self, element):
            """Docstring for map."""
            return {"a": 1}

    class MockDataSource:
        """Docstring for MockDataSource."""

        def __init__(self, ds):
            """Docstring for __init__."""

        def __len__(self):
            """Docstring for __len__."""
            return 10

    with (
        patch("gemma_4_sql.backends.maxtext.etl.datasets", mock_datasets),
        patch("gemma_4_sql.backends.maxtext.etl.grain", mock_grain),
        patch("gemma_4_sql.backends.maxtext.etl._load_hf_or_duckdb", return_value="hf_ds") as mock_load,
        patch("gemma_4_sql.backends.maxtext.etl.get_grain_classes", return_value=(MockDataSource, MockBaseTransform)),
        patch("gemma_4_sql.backends.maxtext.etl.SQLTokenizer", return_value="tokenizer"),
        patch("gemma_4_sql.backends.maxtext.etl._get_sampler", return_value="sampler"),
    ):
        res = build_dataloader(config)
        assert res["dataset"] == "ds"
        assert res["split"] == "split"
        assert res["status"] == "loaded"
        assert res["batch_size"] == 2
        assert res["backend"] == "maxtext"
        assert res["distributed"] is True
        assert res["loader"] == "loader"

        mock_load.assert_called_once_with("ds", "split", "", "")
        mock_grain.DataLoader.assert_called_once()

        # Test the MaxTextFormatTransform
        op = mock_grain.DataLoader.call_args[1]["operations"][0]
        mapped = op.map({"b": 2})
        assert mapped == {"a": 1, "segment_ids": [1], "positions": [0]}


def test_build_dataloader_missing_deps():
    """Docstring for test_build_dataloader_missing_deps."""
    config = ETLConfig(dataset_name="ds", split="split")
    with patch("gemma_4_sql.backends.maxtext.etl.datasets", None), pytest.raises(DependencyMissingError, match="Missing grain or datasets. Cannot load ds."):
        build_dataloader(config)
