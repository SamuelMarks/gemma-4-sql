"""
Tests for LiveDatabaseEngine.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine


def test_live_database_engine_memory() -> None:
    """Test engine with in-memory DB and DDL."""
    ddl = (
        "CREATE TABLE users (id INT, name TEXT); INSERT INTO users VALUES (1, 'Alice');"
    )
    engine = LiveDatabaseEngine(ddl=ddl)

    # Test valid query
    results = engine.execute_query("SELECT * FROM users")
    assert results == [(1, "Alice")]

    # Test compare queries (match)
    assert (
        engine.compare_queries("SELECT id FROM users", "SELECT id FROM users") is True
    )

    # Test compare queries (mismatch)
    assert (
        engine.compare_queries("SELECT id FROM users", "SELECT name FROM users")
        is False
    )

    # Test query that returns no description (like INSERT)
    insert_results = engine.execute_query("INSERT INTO users VALUES (2, 'Bob')")
    assert insert_results == []

    engine.close()


def test_live_database_engine_file() -> None:
    """Test engine with file DB."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        ddl = "CREATE TABLE dummy (val INT); INSERT INTO dummy VALUES (42);"
        engine = LiveDatabaseEngine(db_path=db_path, ddl=ddl)

        # Test error query
        err_results = engine.execute_query("SELECT * FROM non_existent_table")
        assert err_results == []

        engine.close()
    finally:
        os.remove(db_path)


def test_live_database_engine_unsupported() -> None:
    """Test unsupported db_type."""
    with pytest.raises(ValueError, match="Unsupported db_type: mysql"):
        LiveDatabaseEngine(db_type="mysql")


@patch("gemma_4_sql.sdk.db_engine.psycopg2", new=None)
def test_live_database_engine_postgres_missing() -> None:
    """Test PostgreSQL db_type when psycopg2 is missing."""
    with pytest.raises(
        ImportError, match="psycopg2 is required for PostgreSQL support"
    ):
        LiveDatabaseEngine(db_type="postgresql")


@patch("gemma_4_sql.sdk.db_engine.snowflake", new=None)
def test_live_database_engine_snowflake_missing() -> None:
    """Test Snowflake db_type when snowflake-connector-python is missing."""
    with pytest.raises(
        ImportError,
        match="snowflake-connector-python is required for Snowflake support",
    ):
        LiveDatabaseEngine(db_type="snowflake")


def test_live_database_engine_postgres_success() -> None:
    """Test PostgreSQL db_type with a mock psycopg2."""
    mock_psycopg2 = MagicMock()
    mock_conn = MagicMock()
    mock_psycopg2.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.description = [("col",)]
    mock_cursor.fetchall.return_value = [(1,)]

    with patch("gemma_4_sql.sdk.db_engine.psycopg2", new=mock_psycopg2):
        engine = LiveDatabaseEngine(
            db_path="postgresql://user:pass@localhost:5432/db",
            db_type="postgresql",
            ddl="CREATE TABLE p (v INT);",
        )
        assert engine.execute_query("SELECT 1") == [(1,)]
        mock_psycopg2.connect.assert_called_once_with(
            "postgresql://user:pass@localhost:5432/db"
        )
        mock_cursor.execute.assert_any_call("CREATE TABLE p (v INT);")
        mock_conn.commit.assert_called_once()
        engine.close()
        mock_conn.close.assert_called_once()

        # Test error branch
        mock_cursor.execute.side_effect = Exception("DB Error")
        assert engine.execute_query("SELECT 1") == []


def test_live_database_engine_postgres_kwargs_only() -> None:
    """Test PostgreSQL db_type using only db_kwargs (db_path=:memory:)."""
    mock_psycopg2 = MagicMock()
    mock_conn = MagicMock()
    mock_psycopg2.connect.return_value = mock_conn

    with patch("gemma_4_sql.sdk.db_engine.psycopg2", new=mock_psycopg2):
        LiveDatabaseEngine(
            db_path=":memory:",  # Default value, should trigger kwargs only branch
            db_type="postgresql",
            db_kwargs={"host": "localhost"},
        )
        mock_psycopg2.connect.assert_called_once_with(host="localhost")


def test_live_database_engine_snowflake_success() -> None:
    """Test Snowflake db_type with a mock snowflake connector."""
    mock_snowflake = MagicMock()
    mock_conn = MagicMock()
    mock_snowflake.connector.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.description = [("col",)]
    mock_cursor.fetchall.return_value = [(1,)]

    with patch("gemma_4_sql.sdk.db_engine.snowflake", new=mock_snowflake):
        engine = LiveDatabaseEngine(
            db_type="snowflake", db_kwargs={"account": "xy12345", "user": "admin"}
        )
        assert engine.execute_query("SELECT 1") == [(1,)]
        mock_snowflake.connector.connect.assert_called_once_with(
            account="xy12345", user="admin"
        )
        engine.close()
        mock_conn.close.assert_called_once()

def test_execute_with_feedback_sqlite() -> None:
    """Test execute_with_feedback for sqlite."""
    engine = LiveDatabaseEngine(ddl="CREATE TABLE t (id INT); INSERT INTO t VALUES (1);")
    
    # Success with results
    success, res, err = engine.execute_with_feedback("SELECT * FROM t")
    assert success is True
    assert res == [(1,)]
    assert err is None
    
    # Success without results
    success, res, err = engine.execute_with_feedback("INSERT INTO t VALUES (2)")
    assert success is True
    assert res == []
    assert err is None
    
    # Error
    success, res, err = engine.execute_with_feedback("SELECT * FROM non_existent")
    assert success is False
    assert res == []
    assert "no such table: non_existent" in err
    engine.close()

def test_execute_with_feedback_postgres() -> None:
    """Test execute_with_feedback for postgres."""
    mock_psycopg2 = MagicMock()
    mock_conn = MagicMock()
    mock_psycopg2.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    with patch("gemma_4_sql.sdk.db_engine.psycopg2", new=mock_psycopg2):
        engine = LiveDatabaseEngine(db_type="postgresql")
        
        # Success with results
        mock_cursor.description = [("id",)]
        mock_cursor.fetchall.return_value = [(1,)]
        success, res, err = engine.execute_with_feedback("SELECT 1")
        assert success is True
        assert res == [(1,)]
        assert err is None
        
        # Error
        mock_cursor.execute.side_effect = Exception("syntax error")
        success, res, err = engine.execute_with_feedback("SELECT INVALID")
        assert success is False
        assert res == []
        assert "syntax error" in err
        engine.close()

@patch("gemma_4_sql.sdk.db_engine.duckdb", new=None)
def test_live_database_engine_duckdb_missing() -> None:
    """Test DuckDB db_type when duckdb is missing."""
    with pytest.raises(
        ImportError,
        match="duckdb is required for DuckDB support",
    ):
        LiveDatabaseEngine(db_type="duckdb")

def test_live_database_engine_duckdb_success() -> None:
    """Test DuckDB db_type with mock duckdb."""
    from unittest.mock import MagicMock, patch
    mock_duckdb = MagicMock()
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [(42,)]
    with patch('gemma_4_sql.sdk.db_engine.duckdb', mock_duckdb):
        engine = LiveDatabaseEngine(db_type='duckdb', ddl='CREATE TABLE d (v INT);')
        assert engine.execute_query('SELECT * FROM d') == [(42,)]
        mock_conn.execute.side_effect = Exception('error')
        assert engine.execute_query('SELECT * FROM invalid') == []
        mock_conn.execute.side_effect = None
        mock_conn.execute.return_value = mock_cursor
        success, res, err = engine.execute_with_feedback('SELECT * FROM d')
        assert success is True
        assert res == [(42,)]
        mock_conn.execute.side_effect = Exception('error')
        success, res, err = engine.execute_with_feedback('SELECT * FROM d')
        assert success is False
        assert 'error' in err
