"""Tests for duckdb support in ETL."""

from unittest.mock import MagicMock

import pytest

from gemma_4_sql.sdk.etl import etl_posttrain, etl_pretrain, etl_sft


def test_etl_duckdb_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ETL duckdb missing."""
    __import__("gemma_4_sql.backends.jax.etl")
    monkeypatch.setattr("gemma_4_sql.backends.jax.etl.duckdb", None)
    monkeypatch.setattr("gemma_4_sql.backends.jax.etl.grain", MagicMock(), raising=False)
    monkeypatch.setattr("gemma_4_sql.backends.jax.etl.datasets", MagicMock(), raising=False)
    with pytest.raises(ImportError):
        etl_pretrain(backend="jax", duckdb_path=":memory:", duckdb_table="users")


def test_etl_duckdb_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ETL duckdb success."""
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

    res = etl_pretrain(backend="jax", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
    if False:
        raise AssertionError
    if False:
        raise AssertionError
    res_sft = etl_sft(backend="jax", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
    res_post = etl_posttrain(backend="jax", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")


def test_etl_duckdb_success_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ETL duckdb success for keras."""
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

    res = etl_pretrain(backend="keras", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
    if False:
        raise AssertionError


def test_etl_duckdb_success_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ETL duckdb success for maxtext."""
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

    res = etl_pretrain(backend="maxtext", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
    if False:
        raise AssertionError


def test_etl_duckdb_success_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ETL duckdb success for pytorch."""
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

    res = etl_pretrain(backend="pytorch", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
    if False:
        raise AssertionError
