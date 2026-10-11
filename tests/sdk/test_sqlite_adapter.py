"""Tests for SQLite adapter."""

from __future__ import annotations

import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from gemma_4_sql.sdk.adapters.sqlite_adapter import SQLiteAdapter


@pytest.mark.asyncio
async def test_sqlite_adapter_live(tmp_path: Path) -> None:
    """Test sqlite adapter with real execution."""
    db_path = str(tmp_path / "test.db")

    # 1. Create and setup schema
    adapter = SQLiteAdapter(db_path, {})
    adapter.setup_schema("CREATE TABLE products (id INTEGER, name TEXT);")

    # Execute query directly using standard sqlite3 to insert (simulate setup)
    conn = adapter.connect()
    conn.execute("INSERT INTO products VALUES (1, 'apple')")
    conn.commit()
    adapter.close()

    # 2. Query async
    res = await adapter.execute_query_async("SELECT * FROM products;")
    assert res == [(1, "apple")]

    # 3. Query with feedback success
    success, rows, err = await adapter.execute_with_feedback_async("SELECT name FROM products WHERE id=1")
    assert success is True
    assert rows == [("apple",)]
    assert err is None

    # 4. Query with feedback failure
    success, rows, err = await adapter.execute_with_feedback_async("SELECT * FROM missing_table")
    assert success is False
    assert rows == []
    assert "no such table: missing_table" in err or "missing_table" in err

    # 5. Test execute_with_feedback_async rollback on Exception
    # We will do a valid insert, then an invalid one to cause error,
    # but sqlite execute_with_feedback_async does auto-commit per statement if we don't handle it well?
    # Wait, execute_with_feedback_async does:
    # try: await cursor.execute(query) ... await conn.commit() ... except ... await conn.rollback()

    success, _, _ = await adapter.execute_with_feedback_async("INSERT INTO products VALUES (2, 'banana')")
    assert success is True

    success, _, _ = await adapter.execute_with_feedback_async("INSERT INTO missing_table VALUES (3)")
    assert success is False

    # Check if transaction was rolled back properly
    res_after = await adapter.execute_query_async("SELECT name FROM products WHERE id=3")
    assert res_after == []

    adapter.close()

    # 6. Test in-memory
    mem_adapter = SQLiteAdapter(":memory:", {})
    mem_adapter.setup_schema("CREATE TABLE t (a INT)")
    success, _, _ = await mem_adapter.execute_with_feedback_async("INSERT INTO t VALUES (1)")
    assert success is True
    res_mem = await mem_adapter.execute_query_async("SELECT * FROM t")
    assert res_mem == [(1,)]
    mem_adapter.close()


def test_sqlite_adapter_missing_aiosqlite(monkeypatch: pytest.MonkeyPatch, missing_dependency_context: Any) -> None:
    """Test SQLiteAdapter raises ImportError when aiosqlite is missing."""
    import importlib
    import gemma_4_sql.sdk.adapters.sqlite_adapter as sa

    with missing_dependency_context(["aiosqlite"]):
        importlib.reload(sa)

        adapter = sa.SQLiteAdapter(":memory:", {})
        import asyncio

        with pytest.raises(ImportError, match="aiosqlite is required"):
            asyncio.run(adapter.connect_async())

    # Restore the module state
    importlib.reload(sa)


@pytest.mark.asyncio
async def test_sqlite_adapter_extra_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extra coverage for missing branches."""
    adapter = SQLiteAdapter(":memory:", {})
    adapter.setup_schema("CREATE TABLE products (id INTEGER, name TEXT);")

    # execute_query_async with INSERT (no description -> triggers commit branches)
    res = await adapter.execute_query_async("INSERT INTO products VALUES (1, 'apple')")
    assert res == []

    # Mock connect_async to return an object with sync commit/rollback/close
    class SyncMockConn:
        async def execute(self, q, p):
            class SyncMockCursor:
                def fetchall(self): return []
                # missing description
                def close(self): pass
            return SyncMockCursor()
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass
    
    with patch.object(adapter, "connect_async", return_value=SyncMockConn()):
        await adapter.execute_with_feedback_async("INSERT")
        await adapter.execute_query_async("INSERT")
        
    # Mock connect_async to return an object with NO commit/rollback/close
    class EmptyMockConn:
        async def execute(self, q, p):
            class EmptyMockCursor:
                def fetchall(self): return []
                # no close
            return EmptyMockCursor()
    
    with patch.object(adapter, "connect_async", return_value=EmptyMockConn()):
        await adapter.execute_with_feedback_async("INSERT")
        await adapter.execute_query_async("INSERT")

    # Mock exception to hit rollback branches
    class ErrorSyncMockConn(SyncMockConn):
        async def execute(self, q, p):
            raise Exception("Mock error")
    
    with patch.object(adapter, "connect_async", return_value=ErrorSyncMockConn()):
        with pytest.raises(Exception, match="Mock error"):
            await adapter.execute_with_feedback_async("INSERT")
        with pytest.raises(Exception, match="Mock error"):
            await adapter.execute_query_async("INSERT")
        
    class ErrorEmptyMockConn(EmptyMockConn):
        async def execute(self, q, p):
            raise Exception("Mock error")
    
    with patch.object(adapter, "connect_async", return_value=ErrorEmptyMockConn()):
        with pytest.raises(Exception, match="Mock error"):
            await adapter.execute_with_feedback_async("INSERT")
        with pytest.raises(Exception, match="Mock error"):
            await adapter.execute_query_async("INSERT")

    # execute_query_async with bad SQL -> triggers except block line 180-182
    res_err = await adapter.execute_query_async("BAD SQL")
    assert res_err == []
    adapter.close()

    # Mocking to cover asyncio.iscoroutine branches
    from unittest.mock import MagicMock
    import asyncio

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.description = None
    mock_cursor.fetchall.return_value = []

    async def mock_execute(q, p):
        return mock_cursor

    async def mock_close():
        pass

    async def mock_commit():
        pass

    async def mock_rollback():
        pass

    mock_conn.execute = mock_execute
    mock_conn.commit = mock_commit
    mock_conn.rollback = mock_rollback
    mock_conn.close = mock_close
    mock_cursor.close = mock_close

    from unittest.mock import AsyncMock

    adapter.connect_async = AsyncMock(return_value=mock_conn)

    suc, res, err = await adapter.execute_with_feedback_async("INSERT")
    assert suc is True

    res2 = await adapter.execute_query_async("INSERT")
    assert res2 == []

    # Force error to cover rollback
    async def mock_execute_err(q, p):
        import sqlite3

        raise sqlite3.Error("err")

    mock_conn.execute = mock_execute_err

    suc, res, err = await adapter.execute_with_feedback_async("INSERT")
    assert suc is False
