"""
Tests for duckdb support in ETL.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.sdk.etl import etl_pretrain, etl_sft, etl_posttrain


@patch("gemma_4_sql.backends.jax.etl.datasets", new=MagicMock())
@patch("gemma_4_sql.backends.jax.etl.grain", new=MagicMock())
@patch("gemma_4_sql.backends.jax.etl.duckdb", new=None)
def test_etl_duckdb_missing() -> None:
    """Test ETL duckdb missing."""
    with pytest.raises(ImportError, match="duckdb is required for DuckDB support"):
        etl_pretrain(backend="jax", duckdb_path=":memory:", duckdb_table="users")


@patch("gemma_4_sql.backends.jax.etl.datasets", new=MagicMock())
@patch("gemma_4_sql.backends.jax.etl.grain", new=MagicMock())
def test_etl_duckdb_success() -> None:
    """Test ETL duckdb success."""
    import duckdb
    with patch("gemma_4_sql.backends.jax.etl.duckdb", duckdb):
        conn = MagicMock()
        mock_duckdb = MagicMock()
        mock_duckdb.connect.return_value = conn
        
        mock_df = MagicMock()
        mock_df.to_dict.return_value = [{"sql_prompt": "Get users", "sql": "SELECT * FROM users"}]
        
        mock_execute = MagicMock()
        mock_execute.fetchdf.return_value = mock_df
        conn.execute.return_value = mock_execute
        
        with patch("gemma_4_sql.backends.jax.etl.duckdb.connect", mock_duckdb.connect):
            res = etl_pretrain(backend="jax", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
            
            assert res["status"] == "loaded"
            assert res["backend"] == "jax"
            
            # Test sft and posttrain
            res_sft = etl_sft(backend="jax", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
            assert res_sft["status"] == "loaded"
            
            res_post = etl_posttrain(backend="jax", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
            assert res_post["status"] == "loaded"

@patch("gemma_4_sql.backends.keras.etl.datasets", new=MagicMock())
@patch("gemma_4_sql.backends.keras.etl.grain", new=MagicMock())
def test_etl_duckdb_success_keras() -> None:
    """Test ETL duckdb success for keras."""
    import duckdb
    with patch("gemma_4_sql.backends.keras.etl.duckdb", duckdb):
        conn = MagicMock()
        mock_duckdb = MagicMock()
        mock_duckdb.connect.return_value = conn
        mock_df = MagicMock()
        mock_df.to_dict.return_value = [{"sql_prompt": "Get users", "sql": "SELECT * FROM users"}]
        conn.execute.return_value.fetchdf.return_value = mock_df
        
        with patch("gemma_4_sql.backends.keras.etl.duckdb.connect", mock_duckdb.connect):
            res = etl_pretrain(backend="keras", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
            assert res["status"] == "loaded"

@patch("gemma_4_sql.backends.maxtext.etl.datasets", new=MagicMock())
@patch("gemma_4_sql.backends.maxtext.etl.grain", new=MagicMock())
def test_etl_duckdb_success_maxtext() -> None:
    """Test ETL duckdb success for maxtext."""
    import duckdb
    with patch("gemma_4_sql.backends.maxtext.etl.duckdb", duckdb):
        conn = MagicMock()
        mock_duckdb = MagicMock()
        mock_duckdb.connect.return_value = conn
        mock_df = MagicMock()
        mock_df.to_dict.return_value = [{"sql_prompt": "Get users", "sql": "SELECT * FROM users"}]
        conn.execute.return_value.fetchdf.return_value = mock_df
        
        with patch("gemma_4_sql.backends.maxtext.etl.duckdb.connect", mock_duckdb.connect):
            res = etl_pretrain(backend="maxtext", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
            assert res["status"] == "loaded"


@patch("gemma_4_sql.backends.pytorch.etl.datasets", new=MagicMock())
@patch("gemma_4_sql.backends.pytorch.etl.torch", new=MagicMock())
@patch("gemma_4_sql.backends.pytorch.etl.Dataset", new=MagicMock())
@patch("gemma_4_sql.backends.pytorch.etl.DataLoader", new=MagicMock())
def test_etl_duckdb_success_pytorch() -> None:
    """Test ETL duckdb success for pytorch."""
    import duckdb
    with patch("gemma_4_sql.backends.pytorch.etl.duckdb", duckdb):
        conn = MagicMock()
        mock_duckdb = MagicMock()
        mock_duckdb.connect.return_value = conn
        mock_df = MagicMock()
        mock_df.to_dict.return_value = [{"sql_prompt": "Get users", "sql": "SELECT * FROM users"}]
        conn.execute.return_value.fetchdf.return_value = mock_df
        
        with patch("gemma_4_sql.backends.pytorch.etl.duckdb.connect", mock_duckdb.connect):
            res = etl_pretrain(backend="pytorch", duckdb_path=":memory:", duckdb_table="users", tokenizer_name="mock")
            assert res["status"] == "loaded"

