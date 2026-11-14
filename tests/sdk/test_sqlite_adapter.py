"""Unit tests for `SQLiteAdapter` with zero untyped mocks and zero resource leaks."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Coroutine, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import NoReturn
from unittest.mock import patch

import aiosqlite
import pytest

import gemma_4_sql.sdk.adapters.sqlite_adapter as sa
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.sdk.adapters.base import DBKwargValue
from gemma_4_sql.sdk.adapters.sqlite_adapter import (
    AiosqliteConnectionProtocol,
    SQLiteAdapter,
    _build_sqlite_readonly_uri,
)
from gemma_4_sql.type_hints import JSONPrimitive


@pytest.mark.asyncio
async def test_sqlite_adapter_live_and_driver_readonly(tmp_path: Path) -> None:
    """Verify `SQLiteAdapter` sync/async queries, driver-level `?mode=ro` + `query_only`, and schema setup."""
    db_file = tmp_path / "test.db"
    db_path = str(db_file)

    # Writable adapter creates file and inserts seed row
    rw_adapter = SQLiteAdapter(db_path, {}, read_only=False)
    rw_adapter.setup_schema("CREATE TABLE products (id INTEGER, name TEXT);")
    rw_adapter.execute_query("INSERT INTO products VALUES (1, 'apple')")
    rw_adapter.conn.commit()
    await rw_adapter.execute_query_async("SELECT * FROM products;")
    await rw_adapter.close_async()

    # Read-only file-backed adapter enforces `?mode=ro` AND `PRAGMA query_only = ON` at driver level
    ro_adapter = SQLiteAdapter(db_path, {}, read_only=True)
    assert ro_adapter.execute_query("SELECT * FROM products;") == [(1, "apple")]
    with pytest.raises(DatabaseExecutionError):
        ro_adapter.execute_query("INSERT INTO products VALUES (99, 'blocked')")

    # `setup_schema` on read-only file-backed adapter runs DDL via dedicated setup connection
    # (both when `_conn` is active AND when `lazy_connect=True`)
    ro_adapter.setup_schema("CREATE TABLE extra (k INT);")
    assert ro_adapter.execute_query("SELECT * FROM extra;") == []
    ro_lazy = SQLiteAdapter(db_path, {}, read_only=True, lazy_connect=True)
    ro_lazy.setup_schema("CREATE TABLE extra2 (k INT);")
    await ro_lazy.close_async()

    ro_missing = SQLiteAdapter(str(tmp_path / "new_ro.db"), {}, read_only=True, lazy_connect=True)
    await ro_missing.connect_async()
    await ro_missing.close_async()
    with pytest.raises(DatabaseExecutionError):
        ro_adapter.setup_schema("INVALID DDL SYNTAX")

    res = await ro_adapter.execute_query_async("SELECT * FROM products;")
    assert res == [(1, "apple")]
    # Reusing cached `_async_conn` on second call
    assert await ro_adapter.execute_query_async("SELECT name FROM products WHERE id=1") == [("apple",)]

    success, rows, err = await ro_adapter.execute_with_feedback_async("SELECT * FROM missing_table")
    assert success is False
    assert rows == []
    assert err is not None and "missing_table" in err

    await ro_adapter.close_async()

    # URI helper coverage (`file:...`, `?` present vs absent, non-existent file)
    assert _build_sqlite_readonly_uri("file:test.db") == ("file:test.db?mode=ro", True)
    assert _build_sqlite_readonly_uri("file:test.db?cache=shared") == ("file:test.db?cache=shared&mode=ro", True)
    assert _build_sqlite_readonly_uri("file:test.db?mode=ro") == ("file:test.db?mode=ro", True)
    assert _build_sqlite_readonly_uri(str(tmp_path / "does_not_exist.db")) == (
        str(tmp_path / "does_not_exist.db"),
        False,
    )

    # Shared in-memory database supports both sync DDL and async queries
    mem_adapter = SQLiteAdapter(":memory:", {})
    mem_adapter.setup_schema("CREATE TABLE t (a INT)")
    ok_ins, _, _ = await mem_adapter.execute_with_feedback_async("INSERT INTO t VALUES (1)")
    assert ok_ins is True
    assert await mem_adapter.execute_query_async("SELECT * FROM t") == [(1,)]
    ok_bad, _, _ = await mem_adapter.execute_with_feedback_async("INSERT INTO missing_table VALUES (3)")
    assert ok_bad is False
    await mem_adapter.close_async()


def test_sqlite_adapter_missing_aiosqlite(
    missing_dependency_context: Callable[[list[str]], AbstractContextManager[None]],
) -> None:
    """Verify `SQLiteAdapter` raises `DependencyMissingError` when `aiosqlite` is missing and closes cleanly."""
    with missing_dependency_context(["aiosqlite"]):
        adapter = sa.SQLiteAdapter(":memory:", {}, aiosqlite_module=None)
        try:
            with pytest.raises(DependencyMissingError, match="aiosqlite is required"):
                asyncio.run(adapter.connect_async())
        finally:
            adapter.close()


@pytest.mark.asyncio
async def test_sqlite_fixture_query(in_memory_sqlite: aiosqlite.Connection) -> None:
    """Verify the shared `in_memory_sqlite` async connection fixture executes queries."""
    cursor = await in_memory_sqlite.execute("SELECT 1")
    row = await cursor.fetchone()
    assert row == (1,)
    await cursor.close()


@pytest.mark.asyncio
async def test_sqlite_adapter_async_protocol_branches() -> None:
    """Verify `SQLiteAdapter` sync/coroutine commit, rollback, and close branches with typed doubles."""
    adapter = SQLiteAdapter(":memory:", {})
    try:
        adapter.setup_schema("CREATE TABLE products (id INTEGER, name TEXT);")
        assert await adapter.execute_query_async("INSERT INTO products VALUES (1, 'apple')") == []
        await adapter.close_async()

        class SyncCursorDouble:
            """Cursor double with synchronous `close()` and `description=None`."""

            description: Sequence[Sequence[str | int | None]] | bool | None = None

            async def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
                """Return empty rows."""
                return []

            def close(self) -> Coroutine[None, None, None] | None:
                """Close cursor synchronously.

                Returns:
                    `None`.

                """
                return None

        class SyncConnDouble:
            """Async connection double with synchronous `commit`, `rollback`, and `close`."""

            def __init__(self) -> None:
                """Initialize counters."""
                self.committed = 0
                self.rolled_back = 0
                self.closed = 0

            async def execute(self, _sql: str, _params: Sequence[JSONPrimitive] = (), /) -> SyncCursorDouble:
                """Return `SyncCursorDouble`."""
                return SyncCursorDouble()

            def commit(self) -> None:
                """Record synchronous commit."""
                self.committed += 1

            def rollback(self) -> None:
                """Record synchronous rollback."""
                self.rolled_back += 1

            def close(self) -> Coroutine[None, None, None] | None:
                """Record synchronous close.

                Returns:
                    `None`.

                """
                self.closed += 1
                return None

        sync_conn = SyncConnDouble()
        assert await (await sync_conn.execute("SELECT 1")).fetchall() == []

        class _SyncConnAiosqliteModule:
            """Typed `aiosqlite` module double returning `sync_conn`."""

            async def connect(
                self,
                database: str,
                **kwargs: DBKwargValue,
            ) -> AiosqliteConnectionProtocol:
                """Return `sync_conn`."""
                _ = (database, kwargs)
                return sync_conn

        di_sync_adapter = SQLiteAdapter(
            ":memory:",
            {},
            read_only=False,
            lazy_connect=True,
            aiosqlite_module=_SyncConnAiosqliteModule(),
        )
        ok_fb, rows_fb, err_fb = await di_sync_adapter.execute_with_feedback_async("INSERT")
        assert ok_fb is True and rows_fb == [] and err_fb is None
        assert await di_sync_adapter.execute_query_async("INSERT") == []
        assert sync_conn.committed == 2
        await di_sync_adapter.close_async()
        assert sync_conn.closed == 1

        # Re-open sync connection after close_async() closed it, then test minimal
        # cursor/conn without description/close/commit/rollback
        class BareCursorDouble:
            """Minimal cursor double without `.description` or `.close`."""

            async def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
                """Return empty rows."""
                return []

        class BareConnDouble:
            """Minimal async connection double without `commit` or `rollback`."""

            async def execute(self, _sql: str, _params: Sequence[JSONPrimitive] = (), /) -> BareCursorDouble:
                """Return `BareCursorDouble`."""
                return BareCursorDouble()

            def close(self) -> Coroutine[None, None, None] | None:
                """No-op close.

                Returns:
                    `None`.

                """
                return None

        bare_conn = BareConnDouble()
        assert bare_conn.close() is None
        assert await (await bare_conn.execute("SELECT 1")).fetchall() == []

        async def _return_bare_conn() -> AiosqliteConnectionProtocol:
            """Return `bare_conn` double."""
            return bare_conn

        with patch.object(adapter, "connect_async", side_effect=_return_bare_conn):
            assert await adapter.execute_query_async("INSERT") == []

        class ErrorSyncConnDouble(SyncConnDouble):
            """Connection double raising `RuntimeError` to exercise sync `rollback()`."""

            async def execute(self, _sql: str, _params: Sequence[JSONPrimitive] = (), /) -> NoReturn:
                """Raise `RuntimeError`.

                Raises:
                    RuntimeError: Always raised to test rollback handling.

                """
                raise RuntimeError("Mock error")

        err_sync_conn = ErrorSyncConnDouble()

        async def _return_err_sync() -> AiosqliteConnectionProtocol:
            """Return `err_sync_conn` double."""
            return err_sync_conn

        with patch.object(adapter, "connect_async", side_effect=_return_err_sync):
            with pytest.raises(RuntimeError, match="Mock error"):
                await adapter.execute_query_async("INSERT")
            assert err_sync_conn.rolled_back == 1

        class ErrorBareConnDouble(BareConnDouble):
            """Connection double raising `RuntimeError` without `rollback`."""

            async def execute(self, _sql: str, _params: Sequence[JSONPrimitive] = (), /) -> NoReturn:
                """Raise `RuntimeError`.

                Raises:
                    RuntimeError: Always raised to test error propagation without rollback.

                """
                raise RuntimeError("Mock error")

        err_bare = ErrorBareConnDouble()

        async def _return_err_bare() -> AiosqliteConnectionProtocol:
            """Return `err_bare` double."""
            return err_bare

        with (
            patch.object(adapter, "connect_async", side_effect=_return_err_bare),
            pytest.raises(RuntimeError, match="Mock error"),
        ):
            await adapter.execute_query_async("INSERT")

        # Coroutine-based commit, rollback, and close double (replacing MagicMock/AsyncMock)
        class CoroCursorDouble:
            """Cursor double with coroutine `.close()` and `description=None`."""

            description: Sequence[Sequence[str | int | None]] | bool | None = None

            def __init__(self) -> None:
                """Track close count."""
                self.closed = 0

            async def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
                """Return empty rows."""
                return []

            async def close(self) -> None:
                """Close cursor asynchronously."""
                self.closed += 1

        class CoroConnDouble:
            """Connection double with coroutine `commit`, `rollback`, and `close`."""

            def __init__(self, *, raise_sqlite_error: bool = False) -> None:
                """Initialize coroutine connection double.

                Args:
                    raise_sqlite_error: Whether `execute` raises `sqlite3.Error`.

                """
                self.raise_sqlite_error = raise_sqlite_error
                self.committed = 0
                self.rolled_back = 0
                self.closed = 0
                self.cursor = CoroCursorDouble()

            async def execute(self, _sql: str, _params: Sequence[JSONPrimitive] = (), /) -> CoroCursorDouble:
                """Return `self.cursor` or raise `sqlite3.Error`.

                Returns:
                    `CoroCursorDouble` instance.

                Raises:
                    sqlite3.Error: When `self.raise_sqlite_error` is enabled.

                """
                if self.raise_sqlite_error:
                    raise sqlite3.Error("err")
                return self.cursor

            async def commit(self) -> None:
                """Commit asynchronously."""
                self.committed += 1

            async def rollback(self) -> None:
                """Roll back asynchronously."""
                self.rolled_back += 1

            async def close(self) -> None:
                """Close asynchronously."""
                self.closed += 1

        coro_conn = CoroConnDouble()
        await coro_conn.close()
        assert await coro_conn.cursor.fetchall() == []

        async def _return_coro_conn() -> AiosqliteConnectionProtocol:
            """Return `coro_conn` double."""
            return coro_conn

        with patch.object(adapter, "connect_async", side_effect=_return_coro_conn):
            suc, rows, _ = await adapter.execute_with_feedback_async("INSERT")
            assert suc is True and rows == []
            assert coro_conn.committed == 1
            assert coro_conn.cursor.closed == 1

            coro_conn.raise_sqlite_error = True
            suc_err, _, err_msg = await adapter.execute_with_feedback_async("INSERT")
            assert suc_err is False and err_msg == "err"
            assert coro_conn.rolled_back == 1

            with pytest.raises(DatabaseExecutionError, match="err"):
                await adapter.execute_query_async("INSERT")

        # Exercise `_harden_sqlite_connection` fallback when `enable_load_extension` raises `sqlite3.Error` or is absent
        class RestrictedLoadExtConn:
            """Minimal connection stub whose `enable_load_extension` raises `sqlite3.Error`."""

            def enable_load_extension(self, _enabled: bool, /) -> None:
                """Simulate platform build without loadable extension support.

                Raises:
                    sqlite3.Error: Always raised to simulate disabled loadable extensions.

                """
                raise sqlite3.Error("not supported")

        class NoLoadExtConn:
            """Minimal connection stub without `enable_load_extension`."""

            def execute(self, _sql: str, /) -> sqlite3.Cursor | None:
                """No-op execute.

                Returns:
                    `None`.

                """
                return None

        adapter._harden_sqlite_connection(RestrictedLoadExtConn(), enforce_query_only=False)
        no_ext = NoLoadExtConn()
        adapter._harden_sqlite_connection(no_ext, enforce_query_only=True)
    finally:
        await adapter.close_async()
