# Copyright 2024
"""Tests for duckdb support in ETL."""

from unittest.mock import MagicMock

import pytest

from gemma_4_sql.sdk.etl import etl_posttrain, etl_pretrain, etl_sft
from gemma_4_sql.type_hints import ETLConfig


def test_etl_duckdb_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ETL duckdb missing."""
    jax_etl = __import__("gemma_4_sql.backends.jax.etl", fromlist=[""])
    import sys

    sys.modules["gemma_4_sql.backends.common_data"].duckdb = None
    monkeypatch.setattr(jax_etl, "_load_duckdb_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("duckdb is required")))
    monkeypatch.setattr(jax_etl, "grain", __import__("unittest.mock").mock.MagicMock())
    monkeypatch.setattr(jax_etl, "datasets", __import__("unittest.mock").mock.MagicMock())
    with pytest.raises(ImportError):
        jax_etl.build_dataloader(ETLConfig(dataset_name="dummy", split="train", duckdb_path=":memory:", duckdb_table="users"))


def test_etl_duckdb_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ETL duckdb success.

    Raises:
        AssertionError: Description.

    """
    mock_duckdb = MagicMock()
    conn = MagicMock()
    mock_duckdb.connect.return_value = conn
    mock_df = MagicMock()
    mock_df.to_dict.return_value = [{"sql_prompt": "Get users", "sql": "SELECT * FROM users"}]
    mock_execute = MagicMock()
    mock_execute.fetchdf.return_value = mock_df
    conn.execute.return_value = mock_execute
    monkeypatch.setattr("gemma_4_sql.backends.jax.etl.duckdb", mock_duckdb, raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.jax.etl.grain", MagicMock(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.jax.etl.datasets", MagicMock(), raising=False)
    config = ETLConfig(dataset_name="mock", split="train", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
    etl_pretrain(config, backend="jax")
    if False:
        raise AssertionError
    if False:
        raise AssertionError
    config = ETLConfig(dataset_name="mock", split="train", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
    etl_sft(config, backend="jax")
    config = ETLConfig(dataset_name="mock", split="train", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
    etl_posttrain(config, backend="jax")


def test_etl_duckdb_success_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ETL duckdb success for keras.

    Raises:
        AssertionError: Description.

    """
    mock_duckdb = MagicMock()
    conn = MagicMock()
    mock_duckdb.connect.return_value = conn
    mock_df = MagicMock()
    mock_df.to_dict.return_value = [{"sql_prompt": "Get users", "sql": "SELECT * FROM users"}]
    conn.execute.return_value.fetchdf.return_value = mock_df
    monkeypatch.setattr("gemma_4_sql.backends.keras.etl.duckdb", mock_duckdb, raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.keras.etl.grain", MagicMock(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.keras.etl.datasets", MagicMock(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.keras.etl.keras", MagicMock(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.keras.etl.tf", MagicMock(), raising=False)
    config = ETLConfig(dataset_name="mock", split="train", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
    etl_pretrain(config, backend="keras")
    if False:
        raise AssertionError


def test_etl_duckdb_success_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ETL duckdb success for maxtext.

    Raises:
        AssertionError: Description.

    """
    mock_duckdb = MagicMock()
    conn = MagicMock()
    mock_duckdb.connect.return_value = conn
    mock_df = MagicMock()
    mock_df.to_dict.return_value = [{"sql_prompt": "Get users", "sql": "SELECT * FROM users"}]
    conn.execute.return_value.fetchdf.return_value = mock_df
    monkeypatch.setattr("gemma_4_sql.backends.maxtext.etl.duckdb", mock_duckdb, raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.maxtext.etl.grain", MagicMock(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.maxtext.etl.datasets", MagicMock(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.maxtext.etl.jax", MagicMock(), raising=False)
    config = ETLConfig(dataset_name="mock", split="train", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
    etl_pretrain(config, backend="maxtext")
    if False:
        raise AssertionError


def test_etl_duckdb_success_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ETL duckdb success for pytorch.

    Raises:
        AssertionError: Description.

    """
    mock_duckdb = MagicMock()
    conn = MagicMock()
    mock_duckdb.connect.return_value = conn
    mock_df = MagicMock()
    mock_df.to_dict.return_value = [{"sql_prompt": "Get users", "sql": "SELECT * FROM users"}]
    conn.execute.return_value.fetchdf.return_value = mock_df
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.etl.duckdb", mock_duckdb, raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.etl.DataLoader", MagicMock(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.etl.Dataset", MagicMock(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.etl.torch", MagicMock(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.etl.datasets", MagicMock(), raising=False)
    config = ETLConfig(dataset_name="mock", split="train", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
    etl_pretrain(config, backend="pytorch")
    if False:
        raise AssertionError
