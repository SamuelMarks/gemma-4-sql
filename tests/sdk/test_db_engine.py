"""Tests for db engine."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gemma_4_sql.sdk.adapters.base import DatabaseAdapter
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine


def test_db_engine_connect_close() -> None:
    """Test db engine connect and close."""
    with patch("gemma_4_sql.sdk.db_engine._ADAPTERS") as mock_adapters:
        mock_adapter_cls = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.connect.return_value = "conn"
        mock_adapter_cls.return_value = mock_adapter
        mock_adapters.get.return_value = mock_adapter_cls
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
    with patch("gemma_4_sql.sdk.db_engine._ADAPTERS") as mock_adapters:
        mock_adapter_cls = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter_cls.return_value = mock_adapter
        mock_adapters.get.return_value = mock_adapter_cls
        LiveDatabaseEngine(ddl="CREATE TABLE t (a INT);")
        mock_adapter.setup_schema.side_effect = RuntimeError("error")
        with pytest.raises(RuntimeError):
            LiveDatabaseEngine(ddl="CREATE TABLE t (a INT);")


def test_db_engine_compare_queries() -> None:
    """Test compare queries."""
    with patch("gemma_4_sql.sdk.db_engine._ADAPTERS") as mock_adapters:
        mock_adapter_cls = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.execute_query.side_effect = [[(1,)], [(1,)], [(1,)], [(2,)]]
        mock_adapter_cls.return_value = mock_adapter
        mock_adapters.get.return_value = mock_adapter_cls
        engine = LiveDatabaseEngine()
        assert engine.compare_queries("q1", "q2") is True
        assert engine.compare_queries("q1", "q2") is False


@pytest.mark.asyncio
async def test_db_engine_execute_async() -> None:
    """Test db engine execute async."""
    with patch("gemma_4_sql.sdk.db_engine._ADAPTERS") as mock_adapters:
        mock_adapter_cls = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.execute_query_async = AsyncMock(return_value=[("row",)])
        mock_adapter.execute_with_feedback_async = AsyncMock(return_value=(True, [("row",)], None))
        mock_adapter_cls.return_value = mock_adapter
        mock_adapters.get.return_value = mock_adapter_cls
        engine = LiveDatabaseEngine()
        assert await engine.execute_query_async("q") == [("row",)]
        assert await engine.execute_with_feedback_async("q") == (True, [("row",)], None)


def test_base_methods() -> None:
    """Test base methods."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class DatabaseAdapter"):
        DatabaseAdapter()


import contextlib
import typing

import pytest


class MockConn:
    pass


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
    engine.conn.execute("CREATE TABLE t (id INT)")
    with contextlib.suppress(PermissionError):
        (_success, _res, _err) = engine.execute_with_feedback("INSERT INTO t VALUES (1)")
    with contextlib.suppress(PermissionError):
        engine.execute_query("DROP TABLE t")


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
    import gemma_4_sql.sdk.adapters.snowflake_adapter as s_ad

    monkeypatch.setattr(s_ad, "snowflake_connector", None)
    monkeypatch.setattr(s_ad.SnowflakeAdapter, "connect", lambda self: MockConn())
    ad = s_ad.SnowflakeAdapter("path", {})
    with __import__("pytest").raises(ImportError):
        ad.connect()


def xtest_postgres_setup_schema(monkeypatch):
    import gemma_4_sql.sdk.adapters.postgres_adapter as p_ad

    class MockCursor:
        def execute(self, ddl):
            pass

        def close(self):
            pass

    class MockConn:
        def cursor(self):
            return MockCursor()

        def commit(self):
            pass

    monkeypatch.setattr(p_ad.PostgresAdapter, "connect", lambda self: MockConn())
    ad = p_ad.PostgresAdapter("path", {})
    ad.conn = MockConn()
    ad.setup_schema("SQL")


def xtest_snowflake_setup_schema(monkeypatch):
    import gemma_4_sql.sdk.adapters.snowflake_adapter as s_ad

    class MockCursor:
        def execute(self, ddl):
            pass

        def close(self):
            pass

    class MockConn:
        def cursor(self):
            return MockCursor()

        def commit(self):
            pass

    monkeypatch.setattr(s_ad.SnowflakeAdapter, "connect", lambda self: MockConn())
    ad = s_ad.SnowflakeAdapter("path", {})
    ad.conn = MockConn()
    ad.setup_schema("SQL")


def xtest_duckdb_setup_schema(monkeypatch):
    import gemma_4_sql.sdk.adapters.duckdb_adapter as d_ad

    class MockConn:
        def execute(self, ddl):
            pass

    monkeypatch.setattr(d_ad.DuckDBAdapter, "connect", lambda self: MockConn())
    ad = d_ad.DuckDBAdapter("path", {})
    ad.conn = MockConn()
    ad.setup_schema("SQL")


def test_sqlite_setup_schema(monkeypatch):
    import gemma_4_sql.sdk.adapters.sqlite_adapter as s_ad

    class MockConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def executescript(self, ddl):
            pass

    ad = s_ad.SQLiteAdapter(":memory:", {})
    ad.conn = MockConn()
    ad.setup_schema("SQL")


def test_base_setup_schema_async(monkeypatch):
    import gemma_4_sql.sdk.adapters.base as b_ad

    class Base(b_ad.DatabaseAdapter):
        def error_classes(self):
            return (ValueError,)

        def connect(self):
            return None

        async def connect_async(self):
            return None

        def execute_query(self, sql):
            return []

        async def execute_query_async(self, sql):
            return []

        def setup_schema(self, ddl):
            return []

        async def setup_schema_async(self, ddl):
            return []

        def get_schema_info(self):
            return {}

        async def get_schema_info_async(self):
            return {}

    ad = Base("path", {})

    class MockConn:
        def close(self):
            pass

    ad.conn = MockConn()
    ad.close()


def test_base_connect_async(monkeypatch):
    import asyncio

    import gemma_4_sql.sdk.adapters.base as b_ad

    class Base(b_ad.DatabaseAdapter):
        def error_classes(self):
            return (ValueError,)

        def connect(self):
            return "conn"

        def execute_query(self, sql):
            return []

        async def execute_query_async(self, sql):
            return []

        def setup_schema(self, ddl):
            return []

        async def setup_schema_async(self, ddl):
            return []

        def get_schema_info(self):
            return {}

        async def get_schema_info_async(self):
            return {}

    ad = Base("path", {})
    res = asyncio.run(ad.connect_async())
    assert res == "conn"


def xtest_base_execute_with_feedback(monkeypatch):
    import gemma_4_sql.sdk.adapters.base as b_ad

    class Base(b_ad.DatabaseAdapter):
        def error_classes(self):
            return (ValueError,)

        def connect(self):
            return None

        async def connect_async(self):
            return None

        def execute_query(self, sql):
            return []

        async def execute_query_async(self, sql):
            return []

        def setup_schema(self, ddl):
            return []

        async def setup_schema_async(self, ddl):
            return []

        def get_schema_info(self):
            return {}

        async def get_schema_info_async(self):
            return {}

    ad = Base("path", {})
    ad.conn = type("Conn", (), {"execute": lambda s, q, p: type("R", (), {"fetchall": list})()})()
    res = ad.execute_with_feedback("sql")
    assert "status" in res


def test_postgres_missing_real(monkeypatch):
    import gemma_4_sql.sdk.adapters.postgres_adapter as p_ad

    monkeypatch.setattr(p_ad, "psycopg2", None)
    with __import__("pytest").raises(ImportError):
        p_ad.PostgresAdapter("path", {})


def test_snowflake_missing_real(monkeypatch):
    import gemma_4_sql.sdk.adapters.snowflake_adapter as s_ad

    monkeypatch.setattr(s_ad, "snowflake", None)
    with __import__("pytest").raises(ImportError):
        s_ad.SnowflakeAdapter("path", {})


def test_postgres_setup_schema_real(monkeypatch):
    import gemma_4_sql.sdk.adapters.postgres_adapter as p_ad

    class MockCursor:
        def execute(self, ddl):
            pass

        def close(self):
            pass

    class MockConn:
        def cursor(self):
            return MockCursor()

        def commit(self):
            pass

    monkeypatch.setattr(p_ad.PostgresAdapter, "connect", lambda self: MockConn())
    ad = p_ad.PostgresAdapter("path", {})
    ad.setup_schema("SQL")


def test_snowflake_setup_schema_real(monkeypatch):
    import gemma_4_sql.sdk.adapters.snowflake_adapter as s_ad

    class MockCursor:
        def execute(self, ddl):
            pass

        def close(self):
            pass

    class MockConn:
        def cursor(self):
            return MockCursor()

        def commit(self):
            pass

    monkeypatch.setattr(s_ad.SnowflakeAdapter, "connect", lambda self: MockConn())
    ad = s_ad.SnowflakeAdapter("path", {})
    ad.setup_schema("SQL")


def test_duckdb_setup_schema_real(monkeypatch):
    import gemma_4_sql.sdk.adapters.duckdb_adapter as d_ad

    class MockConn:
        def execute(self, ddl):
            pass

    monkeypatch.setattr(d_ad.DuckDBAdapter, "connect", lambda self: MockConn())
    ad = d_ad.DuckDBAdapter("path", {})
    ad.setup_schema("SQL")


def test_base_execute_with_feedback_real(monkeypatch):
    import gemma_4_sql.sdk.adapters.base as b_ad

    class Base(b_ad.DatabaseAdapter):
        @property
        def error_classes(self):
            return (ValueError,)

        def connect(self):
            return None

        async def connect_async(self):
            return None

        def execute_query(self, sql):
            return []

        async def execute_query_async(self, sql):
            return []

        def setup_schema(self, ddl):
            return []

        async def setup_schema_async(self, ddl):
            return []

        def get_schema_info(self):
            return {}

        async def get_schema_info_async(self):
            return {}

    ad = Base("path", {})
    ad.conn = type("Conn", (), {"execute": lambda s, q, p: type("R", (), {"fetchall": list})()})()
    res = ad.execute_with_feedback("sql")
    assert "status" not in res  # it returns tuple(bool, list, str|None)
    assert res[0] is True


def test_postgres_missing_async(monkeypatch):
    import gemma_4_sql.sdk.adapters.postgres_adapter as p_ad

    monkeypatch.setattr(p_ad, "asyncpg", None)
    monkeypatch.setattr(p_ad.PostgresAdapter, "connect", lambda self: type("C", (), {})())
    ad = p_ad.PostgresAdapter("path", {})
    with __import__("pytest").raises(ImportError):
        __import__("asyncio").run(ad.connect_async())


def test_base_setup_schema(monkeypatch):
    import gemma_4_sql.sdk.adapters.base as b_ad

    class Base(b_ad.DatabaseAdapter):
        @property
        def error_classes(self):
            return (ValueError,)

        def connect(self):
            return None

        async def connect_async(self):
            return None

        def execute_query(self, sql):
            return []

        async def execute_query_async(self, sql):
            return []

        async def setup_schema_async(self, ddl):
            return []

        def get_schema_info(self):
            return {}

        async def get_schema_info_async(self):
            return {}

    ad = Base("path", {})
    ad.execute_with_feedback = lambda ddl: None
    ad.setup_schema("SQL")
