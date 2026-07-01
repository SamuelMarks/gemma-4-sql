"""Tests for missing DB Engine coverage."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine


def test_db_engine_insert_no_description() -> object:  # type: ignore[return]
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
    engine = LiveDatabaseEngine(db_path=":memory:", db_type="sqlite", read_only=True)
    engine.conn.execute("CREATE TABLE t (id INT)")

    (success, _res, err) = engine.execute_with_feedback("INSERT INTO t VALUES (1)")
    assert success is False
    assert "Safety Violation" in str(err)

    res2 = engine.execute_query("DROP TABLE t")
    assert res2 == []


def test_db_engine_safety_bypass() -> None:
    engine = LiveDatabaseEngine(db_path=":memory:", db_type="sqlite", read_only=False)
    engine.conn.execute("CREATE TABLE t (id INT)")

    (success, _res, _err) = engine.execute_with_feedback("INSERT INTO t VALUES (1)")
    assert success is True


@pytest.mark.asyncio()
async def test_live_database_engine_duckdb_async() -> None:
    """Test DuckDB async fallback."""
    mock_duckdb = MagicMock()
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [(42,)]

    with patch("gemma_4_sql.sdk.db_engine.duckdb", mock_duckdb):
        engine = LiveDatabaseEngine(db_type="duckdb", db_path=":memory:")
        # Test execute_query_async
        res = await engine.execute_query_async("SELECT 42")
        assert res == [(42,)]

        # Test execute_with_feedback_async
        success, res2, err = await engine.execute_with_feedback_async("SELECT 42")
        assert success is True
        assert res2 == [(42,)]
        assert err is None

        # Test connect_async just returns conn
        aconn = await engine.connect_async()
        assert aconn == mock_conn


@pytest.mark.asyncio()
async def test_live_database_engine_postgres_async() -> None:
    """Test asyncpg for postgres."""

    class MockAsyncpg:
        async def connect(self, *args, **kwargs):
            return MockAsyncConn()

    class MockAsyncConn:
        async def fetch(self, query):
            return [{"col": 42}]

        async def close(self):
            pass

    from gemma_4_sql.sdk import db_engine

    mock_psycopg2 = MagicMock()

    with patch.object(db_engine, "asyncpg", MockAsyncpg()), patch.object(db_engine, "psycopg2", mock_psycopg2):
        engine = LiveDatabaseEngine(db_type="postgresql", db_path="postgres://test")

        # execute_query_async
        res = await engine.execute_query_async("SELECT 42")
        assert res == [(42,)]

        # execute_with_feedback_async
        success, res2, err = await engine.execute_with_feedback_async("SELECT 42")
        assert success is True
        assert res2 == [(42,)]
        assert err is None


@pytest.mark.asyncio()
async def test_live_database_engine_async_unsupported() -> None:
    """Test async for unsupported db."""
    mock_snowflake = MagicMock()
    mock_conn = MagicMock()
    mock_snowflake.connector.connect.return_value = mock_conn
    with patch("gemma_4_sql.sdk.db_engine.snowflake", mock_snowflake):
        engine = LiveDatabaseEngine(db_type="snowflake", db_kwargs={"account": "xy12345", "user": "admin"})
        with pytest.raises(ValueError, match="Async operations not natively supported"):
            await engine.connect_async()


@patch("gemma_4_sql.sdk.db_engine.aiosqlite", new=None)
@pytest.mark.asyncio()
async def test_live_database_engine_aiosqlite_missing() -> None:
    """Test SQLite async when aiosqlite missing."""
    engine = LiveDatabaseEngine(db_type="sqlite")
    with pytest.raises(ImportError, match="aiosqlite is required"):
        await engine.connect_async()


@pytest.mark.asyncio()
async def test_live_database_engine_postgres_async_kwargs() -> None:
    """Test asyncpg for postgres with kwargs only."""

    class MockAsyncpg:
        async def connect(self, *args, **kwargs):
            return MockAsyncConn()

    class MockAsyncConn:
        pass

    from gemma_4_sql.sdk import db_engine

    mock_psycopg2 = MagicMock()
    with patch.object(db_engine, "asyncpg", MockAsyncpg()), patch.object(db_engine, "psycopg2", mock_psycopg2):
        engine = LiveDatabaseEngine(db_type="postgresql", db_path=":memory:", db_kwargs={"host": "localhost"})
        await engine.connect_async()


@pytest.mark.asyncio()
async def test_live_database_engine_sqlite_async_no_description(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockAsyncCursor:
        description = None

        async def fetchall(self):
            return []

        async def close(self):
            pass

    class MockAsyncConn:
        async def execute(self, query):
            return MockAsyncCursor()

        async def close(self):
            pass

    class MockAiosqlite:
        async def connect(self, *args, **kwargs):
            return MockAsyncConn()

    from gemma_4_sql.sdk import db_engine

    monkeypatch.setattr(db_engine, "aiosqlite", MockAiosqlite())
    engine = LiveDatabaseEngine(read_only=False)
    success, res, _err = await engine.execute_with_feedback_async("INSERT")
    assert success is True
    assert res == []
    res2 = await engine.execute_query_async("INSERT")
    assert res2 == []


@pytest.mark.asyncio()
async def test_live_database_engine_sqlite_async_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockAsyncConn:
        async def execute(self, query):
            msg = "Async error"
            raise Exception(msg)  # noqa: TRY002

        async def close(self):
            pass

    class MockAiosqlite:
        async def connect(self, *args, **kwargs):
            return MockAsyncConn()

    from gemma_4_sql.sdk import db_engine

    monkeypatch.setattr(db_engine, "aiosqlite", MockAiosqlite())
    engine = LiveDatabaseEngine(read_only=False)
    success, res, err = await engine.execute_with_feedback_async("SELECT")
    assert success is False
    assert res == []
    assert "Async error" in str(err)
    res2 = await engine.execute_query_async("SELECT")
    assert res2 == []


def test_duckdb_readonly_file() -> None:
    mock_duckdb = MagicMock()
    with patch("gemma_4_sql.sdk.db_engine.duckdb", mock_duckdb):
        engine = LiveDatabaseEngine(db_type="duckdb", db_path="my.db", read_only=True)
        # Should set read_only=True in kwargs
        mock_duckdb.connect.assert_called_with("my.db", read_only=True)
