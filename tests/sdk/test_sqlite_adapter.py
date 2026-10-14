"""Unit tests for the SQLite database adapter."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gemma_4_sql.sdk.adapters.sqlite_adapter as sa
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.sdk.adapters.sqlite_adapter import SQLiteAdapter


@pytest.mark.asyncio
async def test_sqlite_adapter_live(tmp_path: Path) -> None:
    """Verify SQLiteAdapter synchronous and asynchronous query and feedback execution."""
    db_path = str(tmp_path / "test.db")

    adapter = SQLiteAdapter(db_path, {})
    adapter.setup_schema("CREATE TABLE products (id INTEGER, name TEXT);")

    conn = adapter.connect()
    conn.execute("INSERT INTO products VALUES (1, 'apple')")
    conn.commit()
    adapter.close()

    res = await adapter.execute_query_async("SELECT * FROM products;")
    assert res == [(1, "apple")]

    success, rows, err = await adapter.execute_with_feedback_async("SELECT name FROM products WHERE id=1")
    assert success is True
    assert rows == [("apple",)]
    assert err is None

    success, rows, err = await adapter.execute_with_feedback_async("SELECT * FROM missing_table")
    assert success is False
    assert rows == []
    assert err is not None
    assert "missing_table" in err

    success, _, _ = await adapter.execute_with_feedback_async("INSERT INTO products VALUES (2, 'banana')")
    assert success is True

    success, _, _ = await adapter.execute_with_feedback_async("INSERT INTO missing_table VALUES (3)")
    assert success is False

    res_after = await adapter.execute_query_async("SELECT name FROM products WHERE id=3")
    assert res_after == []

    adapter.close()

    mem_adapter = SQLiteAdapter(":memory:", {})
    mem_adapter.setup_schema("CREATE TABLE t (a INT)")
    success, _, _ = await mem_adapter.execute_with_feedback_async("INSERT INTO t VALUES (1)")
    assert success is True
    res_mem = await mem_adapter.execute_query_async("SELECT * FROM t")
    assert res_mem == [(1,)]
    mem_adapter.close()


def test_sqlite_adapter_missing_aiosqlite(monkeypatch: pytest.MonkeyPatch, missing_dependency_context: Any) -> None:
    """Verify SQLiteAdapter raises DependencyMissingError when aiosqlite is missing."""
    with missing_dependency_context(["aiosqlite"]):
        monkeypatch.setattr(sa, "aiosqlite", None)
        adapter = sa.SQLiteAdapter(":memory:", {})
        with pytest.raises(DependencyMissingError, match="aiosqlite is required"):
            asyncio.run(adapter.connect_async())


@pytest.mark.asyncio
async def test_sqlite_fixture_query(in_memory_sqlite: Any) -> None:
    """Verify the shared in_memory_sqlite async connection fixture executes queries."""
    cursor = await in_memory_sqlite.execute("SELECT 1")
    row = await cursor.fetchone()
    assert row == (1,)
    await cursor.close()


@pytest.mark.asyncio
async def test_sqlite_adapter_extra_coverage() -> None:
    """Verify SQLiteAdapter commit, rollback, close coroutine handling, and error raising."""
    adapter = SQLiteAdapter(":memory:", {})
    adapter.setup_schema("CREATE TABLE products (id INTEGER, name TEXT);")

    res = await adapter.execute_query_async("INSERT INTO products VALUES (1, 'apple')")
    assert res == []

    class SyncMockConn:
        """Mock connection with synchronous commit, rollback, and close methods."""

        async def execute(self, q: str, p: tuple[object, ...]) -> Any:
            """Return a cursor with synchronous close."""

            class SyncMockCursor:
                """Mock cursor without description."""

                def close(self) -> None:
                    """Close cursor."""

            return SyncMockCursor()

        def commit(self) -> None:
            """Commit transaction."""

        def rollback(self) -> None:
            """Rollback transaction."""

        def close(self) -> None:
            """Close connection."""

    with patch.object(adapter, "connect_async", return_value=SyncMockConn()):
        await adapter.execute_with_feedback_async("INSERT")
        await adapter.execute_query_async("INSERT")

    class EmptyMockConn:
        """Mock connection without commit, rollback, or close attributes."""

        async def execute(self, q: str, p: tuple[object, ...]) -> Any:
            """Return a minimal cursor."""

            class EmptyMockCursor:
                """Minimal cursor without close or description."""

            return EmptyMockCursor()

    with patch.object(adapter, "connect_async", return_value=EmptyMockConn()):
        await adapter.execute_with_feedback_async("INSERT")
        await adapter.execute_query_async("INSERT")

    class ErrorSyncMockConn(SyncMockConn):
        """Mock connection whose execute raises RuntimeError with sync rollback."""

        async def execute(self, q: str, p: tuple[object, ...]) -> Any:
            """Raise RuntimeError."""
            msg = "Mock error"
            raise RuntimeError(msg)

    with patch.object(adapter, "connect_async", return_value=ErrorSyncMockConn()):
        with pytest.raises(RuntimeError, match="Mock error"):
            await adapter.execute_with_feedback_async("INSERT")
        with pytest.raises(RuntimeError, match="Mock error"):
            await adapter.execute_query_async("INSERT")

    class ErrorEmptyMockConn(EmptyMockConn):
        """Mock connection whose execute raises RuntimeError without rollback attribute."""

        async def execute(self, q: str, p: tuple[object, ...]) -> Any:
            """Raise RuntimeError."""
            msg = "Mock error"
            raise RuntimeError(msg)

    with patch.object(adapter, "connect_async", return_value=ErrorEmptyMockConn()):
        with pytest.raises(RuntimeError, match="Mock error"):
            await adapter.execute_with_feedback_async("INSERT")
        with pytest.raises(RuntimeError, match="Mock error"):
            await adapter.execute_query_async("INSERT")

    with pytest.raises(DatabaseExecutionError):
        await adapter.execute_query_async("BAD SQL")
    adapter.close()

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.description = None
    mock_cursor.fetchall.return_value = []

    async def mock_execute(q: str, p: tuple[object, ...]) -> MagicMock:
        """Return mocked cursor."""
        return mock_cursor

    async def mock_close() -> None:
        """Async close coroutine."""

    async def mock_commit() -> None:
        """Async commit coroutine."""

    async def mock_rollback() -> None:
        """Async rollback coroutine."""

    mock_conn.execute = mock_execute
    mock_conn.commit = mock_commit
    mock_conn.rollback = mock_rollback
    mock_conn.close = mock_close
    mock_cursor.close = mock_close

    adapter.connect_async = AsyncMock(return_value=mock_conn)  # type: ignore[method-assign]

    suc, rows, _ = await adapter.execute_with_feedback_async("INSERT")
    assert suc is True
    assert rows == []

    res2 = await adapter.execute_query_async("INSERT")
    assert res2 == []

    async def mock_execute_err(q: str, p: tuple[object, ...]) -> MagicMock:
        """Raise sqlite3.Error to exercise async rollback."""
        msg = "err"
        raise sqlite3.Error(msg)

    mock_conn.execute = mock_execute_err

    suc, _, err_msg = await adapter.execute_with_feedback_async("INSERT")
    assert suc is False
    assert err_msg == "err"
