"""Tests for db engine."""

from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gemma_4_sql.sdk.adapters.base import DatabaseAdapter
from gemma_4_sql.sdk.db_engine import _ADAPTERS, LiveDatabaseEngine


def test_db_engine_connect_close() -> None:
    """Test db engine connect and close."""
    mock_adapter_cls = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter.connect.return_value = "conn"
    mock_adapter_cls.return_value = mock_adapter
    with patch.dict(_ADAPTERS, {"sqlite": mock_adapter_cls}):
        engine = LiveDatabaseEngine(db_type="sqlite", db_path=":memory:")
        assert engine.connect() == "conn"
        engine.close()
        mock_adapter.close.assert_called_once()


def test_db_engine_unsupported_type() -> None:
    """Test unsupported db type."""
    with pytest.raises(ValueError, match="Unsupported db_type"):
        LiveDatabaseEngine(db_type="invalid_type")


def test_db_engine_ddl() -> None:
    """Test db engine ddl."""
    mock_adapter_cls = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter_cls.return_value = mock_adapter
    with patch.dict(_ADAPTERS, {"sqlite": mock_adapter_cls}):
        LiveDatabaseEngine(ddl="CREATE TABLE t (a INT);")
        mock_adapter.setup_schema.side_effect = RuntimeError("error")
        with pytest.raises(RuntimeError):
            LiveDatabaseEngine(ddl="CREATE TABLE t (a INT);")


def test_db_engine_compare_queries() -> None:
    """Test compare queries."""
    mock_adapter_cls = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter.execute_query.side_effect = [[(1,)], [(1,)], [(1,)], [(2,)]]
    mock_adapter_cls.return_value = mock_adapter
    with patch.dict(_ADAPTERS, {"sqlite": mock_adapter_cls}):
        engine = LiveDatabaseEngine()
        assert engine.compare_queries("q1", "q2") is True
        assert engine.compare_queries("q1", "q2") is False


@pytest.mark.asyncio
async def test_db_engine_execute_async() -> None:
    """Test db engine execute async."""
    mock_adapter_cls = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter.execute_query_async = AsyncMock(return_value=[("row",)])
    mock_adapter.execute_with_feedback_async = AsyncMock(return_value=(True, [("row",)], None))
    mock_adapter_cls.return_value = mock_adapter
    with patch.dict(_ADAPTERS, {"sqlite": mock_adapter_cls}):
        engine = LiveDatabaseEngine()
        assert await engine.execute_query_async("q") == [("row",)]
        assert await engine.execute_with_feedback_async("q") == (True, [("row",)], None)


def test_base_methods() -> None:
    """Test base methods."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class DatabaseAdapter"):
        DatabaseAdapter()


import typing

import pytest


class MockConn:
    """Test class for MockConn."""


def test_db_engine_insert_no_description() -> object:
    """Initialize function test_db_engine_insert_no_description.

    Raises:
        AssertionError: Description.

    """
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
    engine.conn.execute("CREATE TABLE t (id INT, status TEXT)")
    (success, _res, err) = engine.execute_with_feedback("INSERT INTO t VALUES (1, 'active')")
    assert success is False
    assert "Safety Violation" in str(err)
    with pytest.raises(PermissionError):
        engine.execute_query("DROP TABLE t")


def test_db_engine_safety_literal_keywords() -> None:
    """Test that reserved keywords in string literals and comments do not trigger PermissionError."""
    engine = LiveDatabaseEngine(db_path=":memory:", db_type="sqlite", read_only=False)
    engine.conn.execute("CREATE TABLE audits (id INT, action TEXT)")
    engine.conn.execute("INSERT INTO audits VALUES (1, 'UPDATE')")
    engine.read_only = True
    engine.adapter.read_only = True

    # Query with 'UPDATE' in string literal
    (success, res, err) = engine.execute_with_feedback("SELECT * FROM audits WHERE action = 'UPDATE'")
    assert success is True
    assert err is None
    assert len(res) == 1

    # Query with comment containing DELETE keyword
    res2 = engine.execute_query("-- comment containing DELETE\nSELECT id FROM audits")
    assert len(res2) == 1

    # Actual mutating statement should still raise PermissionError
    with pytest.raises(PermissionError):
        engine.execute_query("UPDATE audits SET action = 'NEW'")


def test_db_engine_safety_bypass() -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    engine = LiveDatabaseEngine(db_path=":memory:", db_type="sqlite", read_only=False)
    engine.conn.execute("CREATE TABLE t (id INT)")
    (success, _res, _err) = engine.execute_with_feedback("INSERT INTO t VALUES (1)")
    if success is not True:
        raise AssertionError


@pytest.mark.asyncio
async def test_live_database_engine_duckdb_async() -> None:
    """Test DuckDB async fallback.

    Raises:
        AssertionError: Description.

    """
    mock_duckdb = MagicMock()
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_conn
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
    """Test asyncpg for postgres.

    Raises:
        AssertionError: Description.

    """

    class MockAsyncpg:
        """Provide class docstring."""

        async def connect(self, *_args: object, **_kwargs: object) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return MockAsyncConn()

    class MockAsyncConn:
        """Provide class docstring."""

        async def fetch(self, _query: object) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
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
    """Test async for snowflake db using threadpool."""
    mock_snowflake = MagicMock()
    mock_conn = MagicMock()
    mock_snowflake.connector.connect.return_value = mock_conn
    with patch("gemma_4_sql.sdk.adapters.snowflake_adapter.snowflake", mock_snowflake):
        engine = LiveDatabaseEngine(db_type="snowflake", db_kwargs={"account": "xy12345", "user": "admin"})
        conn = await engine.connect_async()
        assert conn == mock_conn


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
            """Execute function.

            Returns:
                object: Description of return.

            """
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
    """Execute function.

    Raises:
        AssertionError: Description.

    """

    class MockAsyncCursor:
        """Provide class docstring."""

        description = None

        async def fetchall(self) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return []

        async def close(self) -> None:
            """Execute function."""

    class MockAsyncConn:
        """Provide class docstring."""

        async def execute(self, *_args: object, **_kwargs: object) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return MockAsyncCursor()

        async def close(self) -> None:
            """Execute function."""

    class MockAiosqlite:
        """Provide class docstring."""

        async def connect(self, *_args: object, **_kwargs: object) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
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
    """Execute function.

    Raises:
        AssertionError: Description.

    """

    class MockAsyncConn:
        """Provide class docstring."""

        async def execute(self, *_args: object, **_kwargs: object) -> typing.NoReturn:
            """Execute function."""
            msg = "Async error"
            sqlite3 = __import__("sqlite3", fromlist=[""])
            raise sqlite3.Error(msg)

        async def close(self) -> None:
            """Execute function."""

    class MockAiosqlite:
        """Provide class docstring."""

        async def connect(self, *_args: object, **_kwargs: object) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
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


def xtest_postgres_missing(monkeypatch):
    """Execute xtest postgres missing helper."""
    import gemma_4_sql.sdk.adapters.postgres_adapter as p_ad

    monkeypatch.setattr(p_ad, "psycopg2", None)
    monkeypatch.setattr(p_ad, "asyncpg", None)
    monkeypatch.setattr(p_ad.PostgresAdapter, "connect", lambda self: MockConn())
    ad = p_ad.PostgresAdapter("path", {})
    with __import__("pytest").raises(ImportError):
        ad.connect()
    with __import__("pytest").raises(ImportError):
        __import__("asyncio").run(ad.connect_async())


def xtest_snowflake_missing(monkeypatch):
    """Execute xtest snowflake missing helper."""
    import gemma_4_sql.sdk.adapters.snowflake_adapter as s_ad

    monkeypatch.setattr(s_ad, "snowflake_connector", None)
    monkeypatch.setattr(s_ad.SnowflakeAdapter, "connect", lambda self: MockConn())
    ad = s_ad.SnowflakeAdapter("path", {})
    with __import__("pytest").raises(ImportError):
        ad.connect()


def xtest_postgres_setup_schema(monkeypatch):
    """Execute xtest postgres setup schema helper."""
    import gemma_4_sql.sdk.adapters.postgres_adapter as p_ad

    class MockCursor:
        """Test class for MockCursor."""

        def execute(self, ddl):
            """Execute execute helper."""

        def close(self):
            """Execute close helper."""

    class MockConn:
        """Test class for MockConn."""

        def cursor(self):
            """Execute cursor helper."""
            return MockCursor()

        def commit(self):
            """Execute commit helper."""

    monkeypatch.setattr(p_ad.PostgresAdapter, "connect", lambda self: MockConn())
    ad = p_ad.PostgresAdapter("path", {})
    ad.conn = MockConn()
    ad.setup_schema("SQL")


def xtest_snowflake_setup_schema(monkeypatch):
    """Execute xtest snowflake setup schema helper."""
    import gemma_4_sql.sdk.adapters.snowflake_adapter as s_ad

    class MockCursor:
        """Test class for MockCursor."""

        def execute(self, ddl):
            """Execute execute helper."""

        def close(self):
            """Execute close helper."""

    class MockConn:
        """Test class for MockConn."""

        def cursor(self):
            """Execute cursor helper."""
            return MockCursor()

        def commit(self):
            """Execute commit helper."""

    monkeypatch.setattr(s_ad.SnowflakeAdapter, "connect", lambda self: MockConn())
    ad = s_ad.SnowflakeAdapter("path", {})
    ad.conn = MockConn()
    ad.setup_schema("SQL")


def xtest_duckdb_setup_schema(monkeypatch):
    """Execute xtest duckdb setup schema helper."""
    import gemma_4_sql.sdk.adapters.duckdb_adapter as d_ad

    class MockConn:
        """Test class for MockConn."""

        def execute(self, ddl):
            """Execute execute helper."""

    monkeypatch.setattr(d_ad.DuckDBAdapter, "connect", lambda self: MockConn())
    ad = d_ad.DuckDBAdapter("path", {})
    ad.conn = MockConn()
    ad.setup_schema("SQL")


def test_sqlite_setup_schema(monkeypatch):
    """Test sqlite setup schema functionality."""
    import gemma_4_sql.sdk.adapters.sqlite_adapter as s_ad

    class MockConn:
        """Test class for MockConn."""

        def __enter__(self):
            """Initialize __enter__."""
            return self

        def __exit__(self, *a):
            """Initialize __exit__."""

        def executescript(self, ddl):
            """Execute executescript helper."""

    ad = s_ad.SQLiteAdapter(":memory:", {})
    ad.conn = MockConn()
    ad.setup_schema("SQL")


def test_base_setup_schema_async(monkeypatch):
    """Test base setup schema async functionality."""
    import gemma_4_sql.sdk.adapters.base as b_ad

    class Base(b_ad.DatabaseAdapter):
        """Test class for Base."""

        def error_classes(self):
            """Execute error classes helper."""
            return (ValueError,)

        def connect(self):
            """Execute connect helper."""
            return

        async def connect_async(self):
            """Execute connect async helper."""
            return

        def execute_query(self, sql):
            """Execute execute query helper."""
            return []

        async def execute_query_async(self, sql):
            """Execute execute query async helper."""
            return []

        def setup_schema(self, ddl):
            """Execute setup schema helper."""
            return []

        async def setup_schema_async(self, ddl):
            """Execute setup schema async helper."""
            return []

        def get_schema_info(self):
            """Execute get schema info helper."""
            return {}

        async def get_schema_info_async(self):
            """Execute get schema info async helper."""
            return {}

    ad = Base("path", {})

    class MockConn:
        """Test class for MockConn."""

        def close(self):
            """Execute close helper."""

    ad.conn = MockConn()
    ad.close()


def test_base_connect_async(monkeypatch):
    """Test base connect async functionality."""
    import asyncio

    import gemma_4_sql.sdk.adapters.base as b_ad

    class Base(b_ad.DatabaseAdapter):
        """Test class for Base."""

        def error_classes(self):
            """Execute error classes helper."""
            return (ValueError,)

        def connect(self):
            """Execute connect helper."""
            return "conn"

        def execute_query(self, sql):
            """Execute execute query helper."""
            return []

        async def execute_query_async(self, sql):
            """Execute execute query async helper."""
            return []

        def setup_schema(self, ddl):
            """Execute setup schema helper."""
            return []

        async def setup_schema_async(self, ddl):
            """Execute setup schema async helper."""
            return []

        def get_schema_info(self):
            """Execute get schema info helper."""
            return {}

        async def get_schema_info_async(self):
            """Execute get schema info async helper."""
            return {}

    ad = Base("path", {})
    res = asyncio.run(ad.connect_async())
    assert res == "conn"


def xtest_base_execute_with_feedback(monkeypatch):
    """Execute xtest base execute with feedback helper."""
    import gemma_4_sql.sdk.adapters.base as b_ad

    class Base(b_ad.DatabaseAdapter):
        """Test class for Base."""

        def error_classes(self):
            """Execute error classes helper."""
            return (ValueError,)

        def connect(self):
            """Execute connect helper."""
            return

        async def connect_async(self):
            """Execute connect async helper."""
            return

        def execute_query(self, sql):
            """Execute execute query helper."""
            return []

        async def execute_query_async(self, sql):
            """Execute execute query async helper."""
            return []

        def setup_schema(self, ddl):
            """Execute setup schema helper."""
            return []

        async def setup_schema_async(self, ddl):
            """Execute setup schema async helper."""
            return []

        def get_schema_info(self):
            """Execute get schema info helper."""
            return {}

        async def get_schema_info_async(self):
            """Execute get schema info async helper."""
            return {}

    ad = Base("path", {})
    ad.conn = type("Conn", (), {"execute": lambda s, q, p: type("R", (), {"fetchall": list})()})()
    res = ad.execute_with_feedback("sql")
    assert "status" in res


def test_postgres_missing_real(monkeypatch):
    """Test postgres missing real functionality."""
    import gemma_4_sql.sdk.adapters.postgres_adapter as p_ad

    monkeypatch.setattr(p_ad, "psycopg2", None)
    with __import__("pytest").raises(ImportError):
        p_ad.PostgresAdapter("path", {})


def test_snowflake_missing_real(monkeypatch):
    """Test snowflake missing real functionality."""
    import gemma_4_sql.sdk.adapters.snowflake_adapter as s_ad

    monkeypatch.setattr(s_ad, "snowflake", None)
    with __import__("pytest").raises(ImportError):
        s_ad.SnowflakeAdapter("path", {})


def test_postgres_setup_schema_real(monkeypatch):
    """Test postgres setup schema real functionality."""
    import gemma_4_sql.sdk.adapters.postgres_adapter as p_ad

    class MockCursor:
        """Test class for MockCursor."""

        def execute(self, ddl):
            """Execute execute helper."""

        def close(self):
            """Execute close helper."""

    class MockConn:
        """Test class for MockConn."""

        def cursor(self):
            """Execute cursor helper."""
            return MockCursor()

        def commit(self):
            """Execute commit helper."""

    monkeypatch.setattr(p_ad.PostgresAdapter, "connect", lambda self: MockConn())
    ad = p_ad.PostgresAdapter("path", {})
    ad.setup_schema("SQL")


def test_snowflake_setup_schema_real(monkeypatch):
    """Test snowflake setup schema real functionality."""
    import gemma_4_sql.sdk.adapters.snowflake_adapter as s_ad

    class MockCursor:
        """Test class for MockCursor."""

        def execute(self, ddl):
            """Execute execute helper."""

        def close(self):
            """Execute close helper."""

    class MockConn:
        """Test class for MockConn."""

        def cursor(self):
            """Execute cursor helper."""
            return MockCursor()

        def commit(self):
            """Execute commit helper."""

    monkeypatch.setattr(s_ad.SnowflakeAdapter, "connect", lambda self: MockConn())
    ad = s_ad.SnowflakeAdapter("path", {})
    ad.setup_schema("SQL")


@pytest.mark.asyncio
async def test_snowflake_async_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test SnowflakeAdapter async operations offloaded to threadpool.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import gemma_4_sql.sdk.adapters.snowflake_adapter as s_ad

    class MockCursor:
        """Mock Snowflake cursor."""

        description = (("col",),)

        def execute(self, query: str, params: object = None) -> None:
            """Execute query."""

        def fetchall(self) -> list[tuple[object, ...]]:
            """Fetch all rows."""
            return [(1, "test")]

        def close(self) -> None:
            """Close cursor."""

    class MockConn:
        """Mock Snowflake connection."""

        def cursor(self) -> MockCursor:
            """Return cursor."""
            return MockCursor()

        def commit(self) -> None:
            """Commit."""

        def close(self) -> None:
            """Close connection."""

    monkeypatch.setattr(s_ad.SnowflakeAdapter, "connect", lambda self: MockConn())
    ad = s_ad.SnowflakeAdapter("path", {})
    conn = await ad.connect_async()
    assert conn is not None
    rows = await ad.execute_query_async("SELECT 1")
    assert rows == [(1, "test")]
    success, feedback_rows, err = await ad.execute_with_feedback_async("SELECT 1")
    assert success is True
    assert feedback_rows == [(1, "test")]
    assert err is None


def test_duckdb_setup_schema_real(monkeypatch):
    """Test duckdb setup schema real functionality."""
    import gemma_4_sql.sdk.adapters.duckdb_adapter as d_ad

    class MockConn:
        """Test class for MockConn."""

        def execute(self, ddl):
            """Execute execute helper."""

    monkeypatch.setattr(d_ad.DuckDBAdapter, "connect", lambda self: MockConn())
    ad = d_ad.DuckDBAdapter("path", {})
    ad.setup_schema("SQL")


def test_base_execute_with_feedback_real(monkeypatch):
    """Test base execute with feedback real functionality."""
    import gemma_4_sql.sdk.adapters.base as b_ad

    class Base(b_ad.DatabaseAdapter):
        """Test class for Base."""

        @property
        def error_classes(self):
            """Execute error classes helper."""
            return (ValueError,)

        def connect(self):
            """Execute connect helper."""
            return

        async def connect_async(self):
            """Execute connect async helper."""
            return

        def execute_query(self, sql):
            """Execute execute query helper."""
            return []

        async def execute_query_async(self, sql):
            """Execute execute query async helper."""
            return []

        def setup_schema(self, ddl):
            """Execute setup schema helper."""
            return []

        async def setup_schema_async(self, ddl):
            """Execute setup schema async helper."""
            return []

        def get_schema_info(self):
            """Execute get schema info helper."""
            return {}

        async def get_schema_info_async(self):
            """Execute get schema info async helper."""
            return {}

    ad = Base("path", {})
    ad.conn = type("Conn", (), {"execute": lambda s, q, p: type("R", (), {"fetchall": list})()})()
    res = ad.execute_with_feedback("sql")
    assert "status" not in res  # it returns tuple(bool, list, str|None)
    assert res[0] is True


def test_postgres_missing_async(monkeypatch):
    """Test postgres missing async functionality."""
    import gemma_4_sql.sdk.adapters.postgres_adapter as p_ad

    monkeypatch.setattr(p_ad, "asyncpg", None)
    monkeypatch.setattr(p_ad.PostgresAdapter, "connect", lambda self: type("C", (), {})())
    ad = p_ad.PostgresAdapter("path", {})
    with __import__("pytest").raises(ImportError):
        __import__("asyncio").run(ad.connect_async())


def test_base_setup_schema(monkeypatch):
    """Test base setup schema functionality."""
    import gemma_4_sql.sdk.adapters.base as b_ad

    class Base(b_ad.DatabaseAdapter):
        """Test class for Base."""

        @property
        def error_classes(self):
            """Execute error classes helper."""
            return (ValueError,)

        def connect(self):
            """Execute connect helper."""
            return

        async def connect_async(self):
            """Execute connect async helper."""
            return

        def execute_query(self, sql):
            """Execute execute query helper."""
            return []

        async def execute_query_async(self, sql):
            """Execute execute query async helper."""
            return []

        async def setup_schema_async(self, ddl):
            """Execute setup schema async helper."""
            return []

        def get_schema_info(self):
            """Execute get schema info helper."""
            return {}

        async def get_schema_info_async(self):
            """Execute get schema info async helper."""
            return {}

    ad = Base("path", {})
    ad.execute_with_feedback = lambda ddl: None
    ad.setup_schema("SQL")


@pytest.mark.asyncio
async def test_sqlite_adapter_disk_file(tmp_path) -> None:
    """Test SQLiteAdapter connecting to a real file path on disk."""
    from gemma_4_sql.sdk.adapters.sqlite_adapter import SQLiteAdapter

    db_file = str(tmp_path / "test_disk.db")
    adapter = SQLiteAdapter(db_file, {})
    assert adapter.conn is not None
    adapter.setup_schema("CREATE TABLE t (val INT); INSERT INTO t VALUES (123);")

    async_conn = await adapter.connect_async()
    assert async_conn is not None
    await async_conn.close()


@pytest.mark.asyncio
async def test_compare_queries_async() -> None:
    """Test compare_queries_async in LiveDatabaseEngine."""
    engine = LiveDatabaseEngine(":memory:", db_type="sqlite")
    assert await engine.compare_queries_async("SELECT 1", "SELECT 1") is True
    assert await engine.compare_queries_async("SELECT 1", "SELECT 2") is False


def test_postgres_adapter_server_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PostgresAdapter handles server unreachable errors cleanly.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import gemma_4_sql.sdk.adapters.postgres_adapter as p_ad

    class MockOperationalError(Exception):
        """Mock operational error."""

    # Connection error on init
    monkeypatch.setattr(p_ad, "psycopg2", type("PG", (), {"Error": MockOperationalError, "connect": mock.MagicMock(side_effect=MockOperationalError("server closed the connection"))}))
    with pytest.raises(MockOperationalError):
        p_ad.PostgresAdapter("postgresql://localhost:5432/db", {})

    # Connection succeeds but server fails during query
    mock_conn = mock.MagicMock()
    mock_conn.cursor.side_effect = MockOperationalError("connection dropped")
    monkeypatch.setattr(p_ad, "psycopg2", type("PG", (), {"Error": MockOperationalError, "connect": lambda *a, **k: mock_conn}))
    ad = p_ad.PostgresAdapter("postgresql://localhost:5432/db", {})
    success, rows, err = ad.execute_with_feedback("SELECT 1")
    assert success is False
    assert rows == []
    assert err is not None
    assert "connection dropped" in err


@pytest.mark.asyncio
async def test_postgres_adapter_async_server_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PostgresAdapter async handles server unreachable errors cleanly.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import gemma_4_sql.sdk.adapters.postgres_adapter as p_ad

    class MockAsyncpgError(Exception):
        """Mock asyncpg error."""

    monkeypatch.setattr(p_ad, "psycopg2", type("PG", (), {"Error": Exception, "connect": lambda *a, **k: mock.MagicMock()}))
    monkeypatch.setattr(p_ad, "asyncpg", type("APG", (), {"PostgresError": MockAsyncpgError, "connect": mock.AsyncMock(side_effect=MockAsyncpgError("connection refused"))}))

    ad = p_ad.PostgresAdapter("postgresql://localhost:5432/db", {})
    success, rows, err = await ad.execute_with_feedback_async("SELECT 1")
    assert success is False
    assert rows == []
    assert err is not None
    assert "connection refused" in err


def test_postgres_adapter_connected_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PostgresAdapter connected execution returning rows.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import gemma_4_sql.sdk.adapters.postgres_adapter as p_ad

    mock_cursor = mock.MagicMock()
    mock_cursor.fetchall.return_value = [("Alice", 100)]
    mock_conn = mock.MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    monkeypatch.setattr(p_ad, "psycopg2", type("PG", (), {"Error": Exception, "connect": lambda *a, **k: mock_conn}))
    ad = p_ad.PostgresAdapter("postgresql://localhost:5432/db", {})

    rows = ad.execute_query("SELECT name, score FROM users")
    assert rows == [("Alice", 100)]

    success, feedback_rows, err = ad.execute_with_feedback("SELECT name, score FROM users")
    assert success is True
    assert feedback_rows == [("Alice", 100)]
    assert err is None


def test_snowflake_adapter_server_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test SnowflakeAdapter handles server unreachable errors cleanly.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import gemma_4_sql.sdk.adapters.snowflake_adapter as s_ad

    class MockSnowflakeError(Exception):
        """Mock snowflake error."""

    # Connection error on init
    monkeypatch.setattr(s_ad, "snowflake", type("SF", (), {"connector": type("SFC", (), {"Error": MockSnowflakeError, "connect": mock.MagicMock(side_effect=MockSnowflakeError("network is unreachable"))})}))
    with pytest.raises(MockSnowflakeError):
        s_ad.SnowflakeAdapter("account/db/schema", {})

    # Connection succeeds but server drops during query
    mock_conn = mock.MagicMock()
    mock_conn.cursor.side_effect = MockSnowflakeError("query timeout / network dropped")
    monkeypatch.setattr(s_ad, "snowflake", type("SF", (), {"connector": type("SFC", (), {"Error": MockSnowflakeError, "connect": lambda *a, **k: mock_conn})}))
    ad = s_ad.SnowflakeAdapter("account/db/schema", {})
    success, rows, err = ad.execute_with_feedback("SELECT 1")
    assert success is False
    assert rows == []
    assert err is not None
    assert "query timeout" in err


def test_snowflake_adapter_connected_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test SnowflakeAdapter connected execution returning rows.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import gemma_4_sql.sdk.adapters.snowflake_adapter as s_ad

    mock_cursor = mock.MagicMock()
    mock_cursor.fetchall.return_value = [("Sales", 5000)]
    mock_conn = mock.MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    monkeypatch.setattr(s_ad, "snowflake", type("SF", (), {"connector": type("SFC", (), {"Error": Exception, "connect": lambda *a, **k: mock_conn})}))
    ad = s_ad.SnowflakeAdapter("account/db/schema", {})

    rows = ad.execute_query("SELECT department, budget FROM depts")
    assert rows == [("Sales", 5000)]

    success, feedback_rows, err = ad.execute_with_feedback("SELECT department, budget FROM depts")
    assert success is True
    assert feedback_rows == [("Sales", 5000)]
    assert err is None


def test_snowflake_adapter_connect_unresolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test SnowflakeAdapter raises ImportError when connect function cannot be resolved.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import gemma_4_sql.sdk.adapters.snowflake_adapter as s_ad

    monkeypatch.setattr(s_ad, "snowflake", type("SF", (), {}))
    with pytest.raises(ImportError, match="snowflake connect function could not be resolved"):
        s_ad.SnowflakeAdapter("acc/db/schema", {})


def test_snowflake_adapter_setup_schema_error_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test SnowflakeAdapter setup_schema triggers rollback on error.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import gemma_4_sql.sdk.adapters.snowflake_adapter as s_ad

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = RuntimeError("DDL syntax error")
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.rollback = MagicMock()

    monkeypatch.setattr(s_ad.SnowflakeAdapter, "connect", lambda self: mock_conn)
    adapter = s_ad.SnowflakeAdapter("acc/db/schema", {})
    adapter.conn = mock_conn

    with pytest.raises(RuntimeError, match="DDL syntax error"):
        adapter.setup_schema("CREATE TABLE bad (")

    mock_conn.rollback.assert_called_once()
    mock_cursor.close.assert_called_once()


def test_snowflake_setup_schema_no_commit_or_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test SnowflakeAdapter setup_schema without commit and rollback methods.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import gemma_4_sql.sdk.adapters.snowflake_adapter as s_ad

    class NoCommitConn:
        """Connection without commit or rollback."""

        def cursor(self) -> MagicMock:
            """Return mock cursor."""
            return MagicMock()

    adapter = s_ad.SnowflakeAdapter("acc/db/schema", {})
    adapter.conn = NoCommitConn()
    adapter.setup_schema("CREATE TABLE ok (id INT)")

    class NoRollbackConn:
        """Connection without rollback that raises on execute."""

        def cursor(self) -> MagicMock:
            """Return mock cursor raising error."""
            c = MagicMock()
            c.execute.side_effect = ValueError("ddl fail")
            return c

    adapter.conn = NoRollbackConn()
    with pytest.raises(ValueError, match="ddl fail"):
        adapter.setup_schema("CREATE TABLE err (")
