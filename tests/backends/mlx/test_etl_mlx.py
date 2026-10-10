"""Tests for mlx etl."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig


def test_mlx_etl_imports():
    """Test mlx etl imports fallback."""
    with patch.dict(sys.modules, {"datasets": None}):
        if "gemma_4_sql.backends.mlx.etl" in sys.modules:
            del sys.modules["gemma_4_sql.backends.mlx.etl"]
        import gemma_4_sql.backends.mlx.etl as etl_module

        assert etl_module.datasets is None


def test_pad_batch():
    """Test _pad_batch."""
    import gemma_4_sql.backends.mlx.etl as etl_module

    inputs = [[1, 2], [1, 2, 3]]
    targets = [[4], [4, 5]]

    res = etl_module._pad_batch(inputs, targets)

    assert res["inputs"] == [[1, 2, 0], [1, 2, 3]]
    assert res["targets"] == [[4, 0], [4, 5]]


def test_mlx_data_loader():
    """Test MLXDataLoader."""
    import gemma_4_sql.backends.mlx.etl as etl_module

    ds = [{"sql_prompt": "p1", "sql": "s1"}, {"question": "p2", "query": "s2"}, {"sql_prompt": "p3", "sql": "s3"}]

    mock_tok = MagicMock()
    mock_tok.encode.side_effect = lambda x: [len(x)]

    loader = etl_module.MLXDataLoader(ds, mock_tok, bs=2)

    batches = list(loader)
    assert len(batches) == 2
    assert batches[0]["inputs"] == [[2], [2]]  # p1, p2
    assert batches[1]["inputs"] == [[2]]  # p3

    # Test empty ds
    empty_loader = etl_module.MLXDataLoader([], mock_tok, bs=2)
    assert list(empty_loader) == []


def test_load_hf_or_duckdb():
    """Test _load_hf_or_duckdb."""
    import gemma_4_sql.backends.mlx.etl as etl_module

    with patch.object(etl_module, "load_duckdb_dataset") as mock_load_duckdb:
        mock_load_duckdb.return_value = "duckdb_ds"
        res = etl_module._load_hf_or_duckdb("ds", "split", "path", "table")
        assert res == "duckdb_ds"

    mock_datasets = MagicMock()
    etl_module.datasets = mock_datasets
    mock_datasets.load_dataset.return_value = "hf_ds"

    res = etl_module._load_hf_or_duckdb("ds", "split", None, None)
    assert res == "hf_ds"

    etl_module.datasets = None
    with pytest.raises(DependencyMissingError, match="Datasets dependency is missing"):
        etl_module._load_hf_or_duckdb("ds", "split", None, None)


def test_build_dataloader():
    """Test build_dataloader."""
    import gemma_4_sql.backends.mlx.etl as etl_module

    config = ETLConfig(dataset_name="ds", split="train", batch_size=2)

    mock_datasets = MagicMock()
    etl_module.datasets = mock_datasets

    with patch.object(etl_module, "_load_hf_or_duckdb") as mock_load:
        mock_load.return_value = "ds"
        with patch.object(etl_module, "SQLTokenizer") as mock_tok:
            mock_tok.return_value = "tok"

            res = etl_module.build_dataloader(config, duckdb_path="path", duckdb_table="table")

            assert res["dataset"] == "ds"
            assert res["split"] == "train"
            assert res["status"] == "loaded"
            assert res["batch_size"] == 2
            assert res["backend"] == "mlx"
            assert res["distributed"] is False
            assert isinstance(res["loader"], etl_module.MLXDataLoader)

    etl_module.datasets = None
    with pytest.raises(DependencyMissingError, match="Missing datasets"):
        etl_module.build_dataloader(config)
