"""Tests for missing DB Engine coverage."""

import contextlib
import typing
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine


def test_db_engine_insert_no_description() -> object:
    """Initialize function test_db_engine_insert_no_description."""
    engine = LiveDatabaseEngine(db_path=":memory:", db_type="sqlite", read_only=False)
    engine.conn.execute("CREATE TABLE t (id INT)")
    (success, res, err) = engine.execute_with_feedback("INSERT INTO t VALUES (1)")
    if success is not True:
        raise AssertionError
    if not res == []:
        raise AssertionError
    if err is not None:
        raise AssertionError


def test_db_engine_safety() -> None:
    """Execute function."""
    engine = LiveDatabaseEngine(db_path=":memory:", db_type="sqlite", read_only=True)
    engine.conn.execute("CREATE TABLE t (id INT)")
    with contextlib.suppress(PermissionError):
        (_success, _res, _err) = engine.execute_with_feedback("INSERT INTO t VALUES (1)")
    with contextlib.suppress(PermissionError):
        engine.execute_query("DROP TABLE t")


def test_db_engine_safety_bypass() -> None:
    """Execute function."""
    engine = LiveDatabaseEngine(db_path=":memory:", db_type="sqlite", read_only=False)
    engine.conn.execute("CREATE TABLE t (id INT)")
    (success, _res, _err) = engine.execute_with_feedback("INSERT INTO t VALUES (1)")
    if success is not True:
        raise AssertionError


@pytest.mark.asyncio
async def test_live_database_engine_duckdb_async() -> None:
    """Test DuckDB async fallback."""
    mock_duckdb = MagicMock()
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [(42,)]
    with patch("gemma_4_sql.sdk.adapters.duckdb_adapter.duckdb", mock_duckdb):
        engine = LiveDatabaseEngine(db_type="duckdb", db_path=":memory:")
        res = await engine.execute_query_async("SELECT 42")
        if res != [(42,)]:
            raise AssertionError
        (success, res2, err) = await engine.execute_with_feedback_async("SELECT 42")
        if success is not True:
            raise AssertionError
        if res2 != [(42,)]:
            raise AssertionError
        if err is not None:
            raise AssertionError
        aconn = await engine.connect_async()
        if aconn != mock_conn:
            raise AssertionError


@pytest.mark.asyncio
async def test_live_database_engine_postgres_async() -> None:
    """Test asyncpg for postgres."""

    class MockAsyncpg:
        """Provide class docstring."""

        async def connect(self, *_args: object, **_kwargs: object) -> object:
            """Execute function."""
            return MockAsyncConn()

    class MockAsyncConn:
        """Provide class docstring."""

        async def fetch(self, _query: object) -> object:
            """Execute function."""
            return [{"col": 42}]

        async def close(self) -> None:
            """Execute function."""

    mock_psycopg2 = MagicMock()
    postgres_adapter = __import__("gemma_4_sql.sdk.adapters", fromlist=["postgres_adapter"]).postgres_adapter
    with patch.object(postgres_adapter, "asyncpg", MockAsyncpg()), patch.object(postgres_adapter, "psycopg2", mock_psycopg2):
        engine = LiveDatabaseEngine(db_type="postgresql", db_path="postgres://test")
        res = await engine.execute_query_async("SELECT 42")
        if res != [(42,)]:
            raise AssertionError
        (success, res2, err) = await engine.execute_with_feedback_async("SELECT 42")
        if success is not True:
            raise AssertionError
        if res2 != [(42,)]:
            raise AssertionError
        if err is not None:
            raise AssertionError


@pytest.mark.asyncio
async def test_live_database_engine_async_unsupported() -> None:
    """Test async for unsupported db."""
    mock_snowflake = MagicMock()
    mock_conn = MagicMock()
    mock_snowflake.connector.connect.return_value = mock_conn
    with patch("gemma_4_sql.sdk.adapters.snowflake_adapter.snowflake", mock_snowflake):
        engine = LiveDatabaseEngine(db_type="snowflake", db_kwargs={"account": "xy12345", "user": "admin"})
        with pytest.raises(ValueError, match="Async operations not natively supported"):
            await engine.connect_async()


@patch("gemma_4_sql.sdk.adapters.sqlite_adapter.aiosqlite", new=None)
@pytest.mark.asyncio
async def test_live_database_engine_aiosqlite_missing() -> None:
    """Test SQLite async when aiosqlite missing."""
    engine = LiveDatabaseEngine(db_type="sqlite")
    with pytest.raises(ImportError, match="aiosqlite is required"):
        await engine.connect_async()


@pytest.mark.asyncio
async def test_live_database_engine_postgres_async_kwargs() -> None:
    """Test asyncpg for postgres with kwargs only."""

    class MockAsyncpg:
        """Provide class docstring."""

        async def connect(self, *_args: object, **_kwargs: object) -> object:
            """Execute function."""
            return MockAsyncConn()

    class MockAsyncConn:
        """Provide class docstring."""

    mock_psycopg2 = MagicMock()
    postgres_adapter = __import__("gemma_4_sql.sdk.adapters", fromlist=["postgres_adapter"]).postgres_adapter
    with patch.object(postgres_adapter, "asyncpg", MockAsyncpg()), patch.object(postgres_adapter, "psycopg2", mock_psycopg2):
        engine = LiveDatabaseEngine(db_type="postgresql", db_path=":memory:", db_kwargs={"host": "localhost"})
        await engine.connect_async()


@pytest.mark.asyncio
async def test_live_database_engine_sqlite_async_no_description(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""

    class MockAsyncCursor:
        """Provide class docstring."""

        description = None

        async def fetchall(self) -> object:
            """Execute function."""
            return []

        async def close(self) -> None:
            """Execute function."""

    class MockAsyncConn:
        """Provide class docstring."""

        async def execute(self, _query: object) -> object:
            """Execute function."""
            return MockAsyncCursor()

        async def close(self) -> None:
            """Execute function."""

    class MockAiosqlite:
        """Provide class docstring."""

        async def connect(self, *_args: object, **_kwargs: object) -> object:
            """Execute function."""
            return MockAsyncConn()

    sqlite_adapter = __import__("gemma_4_sql.sdk.adapters", fromlist=["sqlite_adapter"]).sqlite_adapter
    monkeypatch.setattr(sqlite_adapter, "aiosqlite", MockAiosqlite())
    engine = LiveDatabaseEngine(read_only=False)
    (success, res, _err) = await engine.execute_with_feedback_async("INSERT")
    if success is not True:
        raise AssertionError
    if res != []:
        raise AssertionError
    await engine.execute_query_async("INSERT")


@pytest.mark.asyncio
async def test_live_database_engine_sqlite_async_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""

    class MockAsyncConn:
        """Provide class docstring."""

        async def execute(self, _query: object) -> typing.Never:
            """Execute function."""
            msg = "Async error"
            sqlite3 = __import__("sqlite3")
            raise sqlite3.Error(msg)

        async def close(self) -> None:
            """Execute function."""

    class MockAiosqlite:
        """Provide class docstring."""

        async def connect(self, *_args: object, **_kwargs: object) -> object:
            """Execute function."""
            return MockAsyncConn()

    sqlite_adapter = __import__("gemma_4_sql.sdk.adapters", fromlist=["sqlite_adapter"]).sqlite_adapter
    monkeypatch.setattr(sqlite_adapter, "aiosqlite", MockAiosqlite())
    engine = LiveDatabaseEngine(read_only=False)
    (_success, res, err) = await engine.execute_with_feedback_async("SELECT")
    if res != []:
        raise AssertionError
    if "Async error" not in str(err):
        raise AssertionError
    await engine.execute_query_async("SELECT")


def test_duckdb_readonly_file() -> None:
    """Execute function."""
    mock_duckdb = MagicMock()
    with patch("gemma_4_sql.sdk.adapters.duckdb_adapter.duckdb", mock_duckdb):
        LiveDatabaseEngine(db_type="duckdb", db_path="my.db", read_only=True)
        mock_duckdb.connect.assert_called_with("my.db", read_only=True)
