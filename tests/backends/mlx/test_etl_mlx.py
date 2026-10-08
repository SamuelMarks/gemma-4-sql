"""Module docstring."""

from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.mlx.etl as mlx_etl
from gemma_4_sql.backends.mlx.etl import MLXDataLoader, _load_hf_or_duckdb, _pad_batch, build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig


def test_pad_batch():
    """Docstring for test_pad_batch."""
    inputs = [[1, 2], [1]]
    targets = [[3], [3, 4, 5]]
    res = _pad_batch(inputs, targets)
    assert res["inputs"] == [[1, 2], [1, 0]]
    assert res["targets"] == [[3, 0, 0], [3, 4, 5]]


def test_mlx_dataloader():
    """Docstring for test_mlx_dataloader."""
    ds = [
        {"sql_prompt": "p1", "sql": "s1"},
        {"question": "p2", "query": "s2"},
        {"sql_prompt": "p3", "sql": "s3"},
    ]
    tok = MagicMock()
    tok.encode.side_effect = lambda x: [len(x)]  # mock encoding

    loader = MLXDataLoader(ds, tok, bs=2)
    batches = list(loader)

    assert len(batches) == 2
    assert batches[0]["inputs"] == [[2], [2]]  # p1, p2
    assert batches[0]["targets"] == [[2], [2]]  # s1, s2
    assert batches[1]["inputs"] == [[2]]  # p3
    assert batches[1]["targets"] == [[2]]  # s3


def test_mlx_dataloader_fallback():
    """Docstring for test_mlx_dataloader_fallback."""
    ds = [{}]
    tok = MagicMock()
    tok.encode.return_value = [1]
    loader = MLXDataLoader(ds, tok, bs=1)
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0]["inputs"] == [[1]]


def test_load_hf_or_duckdb_duckdb(monkeypatch):
    """Docstring for test_load_hf_or_duckdb_duckdb."""
    with patch("gemma_4_sql.backends.mlx.etl.load_duckdb_dataset") as mock_load:
        mock_load.return_value = "duckdb_ds"
        res = _load_hf_or_duckdb("ds", "train", "path", "table")
        assert res == "duckdb_ds"
        mock_load.assert_called_with("path", "table")


def test_load_hf_or_duckdb_hf_missing_deps(monkeypatch):
    """Docstring for test_load_hf_or_duckdb_hf_missing_deps."""
    monkeypatch.setattr(mlx_etl, "datasets", None)
    with pytest.raises(DependencyMissingError, match="Datasets dependency is missing"):
        _load_hf_or_duckdb("ds", "train", None, None)


def test_load_hf_or_duckdb_hf(monkeypatch):
    """Docstring for test_load_hf_or_duckdb_hf."""
    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = "hf_ds"
    monkeypatch.setattr(mlx_etl, "datasets", mock_datasets)

    res = _load_hf_or_duckdb("ds", "train", None, None)
    assert res == "hf_ds"


def test_build_dataloader_missing_deps(monkeypatch):
    """Docstring for test_build_dataloader_missing_deps."""
    monkeypatch.setattr(mlx_etl, "datasets", None)
    with pytest.raises(DependencyMissingError, match="Missing datasets."):
        build_dataloader(ETLConfig(dataset_name="ds", split="train", batch_size=2))


def test_build_dataloader_success(monkeypatch):
    """Docstring for test_build_dataloader_success."""
    mock_datasets = MagicMock()
    monkeypatch.setattr(mlx_etl, "datasets", mock_datasets)

    with patch("gemma_4_sql.backends.mlx.etl._load_hf_or_duckdb") as mock_load:
        mock_load.return_value = ["ds"]
        with patch("gemma_4_sql.backends.mlx.etl.SQLTokenizer"):
            res = build_dataloader(ETLConfig(dataset_name="ds", split="train", batch_size=2), duckdb_path="p", duckdb_table="t")
            assert res["status"] == "loaded"
            assert res["backend"] == "mlx"
            assert res["batch_size"] == 2
            assert res["dataset"] == "ds"
            assert isinstance(res["loader"], MLXDataLoader)
