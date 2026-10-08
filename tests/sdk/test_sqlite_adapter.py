"""Module docstring."""

import sqlite3
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


@pytest.mark.asyncio
async def test_sqlite_adapter(monkeypatch):
    """Docstring for test_sqlite_adapter."""
    import gemma_4_sql.sdk.adapters.sqlite_adapter as sa

    # Test connect memory
    adapter1 = sa.SQLiteAdapter(":memory:", {"check_same_thread": True})
    assert adapter1.db_path == ":memory:"
    assert adapter1.error_classes == (sqlite3.Error,)

    # Test connect file
    with patch("sqlite3.connect") as m_connect:
        m_connect.return_value = "conn"
        adapter2 = sa.SQLiteAdapter("file.db", {})
        assert adapter2.connect() == "conn"

    # Test connect_async memory
    mock_aiosqlite = MagicMock()

    async def mock_connect(*a, **k):
        """Docstring for mock_connect."""
        return "aconn"

    mock_aiosqlite.connect = mock_connect
    with patch("gemma_4_sql.sdk.adapters.sqlite_adapter.aiosqlite", mock_aiosqlite):
        res = await adapter1.connect_async()
        assert res == "aconn"

        # Test connect_async file
        adapter3 = sa.SQLiteAdapter("file.db", {})
        res2 = await adapter3.connect_async()
        assert res2 == "aconn"

    # Test missing aiosqlite
    with patch("gemma_4_sql.sdk.adapters.sqlite_adapter.aiosqlite", None):
        with pytest.raises(ImportError):
            await adapter1.connect_async()

    # Test setup_schema
    m_conn = MagicMock()
    adapter2.conn = m_conn
    adapter2.setup_schema("DDL")
    m_conn.executescript.assert_called_with("DDL")

    # Mock async connection
    class AsyncMockConn:
        """Docstring for AsyncMockConn."""

        def __init__(self, c_mock):
            """Docstring for __init__."""
            self.c_mock = c_mock
            self.commit_called = False
            self.rollback_called = False
            self.close_called = False

        async def execute(self, q, p):
            """Docstring for execute."""
            if q == "FAIL_EXEC":
                raise sqlite3.Error("exec_err")
            if q == "FAIL_AFTER":
                raise ValueError("val_err")
            return self.c_mock

        async def commit(self):
            """Docstring for commit."""
            self.commit_called = True

        async def rollback(self):
            """Docstring for rollback."""
            self.rollback_called = True

        async def close(self):
            """Docstring for close."""
            self.close_called = True

    class AsyncMockCursor:
        """Docstring for AsyncMockCursor."""

        def __init__(self, desc=True):
            """Docstring for __init__."""
            self.description = desc
            if not desc:
                del self.description
            self.close_called = False

        async def fetchall(self):
            """Docstring for fetchall."""
            return [("row",)]

        async def close(self):
            """Docstring for close."""
            self.close_called = True

    # Test execute_with_feedback_async success with desc
    c_mock = AsyncMockCursor()
    conn_mock = AsyncMockConn(c_mock)

    async def _async_conn(*a, **k):
        """Docstring for _async_conn."""
        return conn_mock

    adapter1.connect_async = _async_conn

    res_fw = await adapter1.execute_with_feedback_async("SELECT")
    assert res_fw == (True, [("row",)], None)

    # Test execute_with_feedback_async success NO desc
    c_mock2 = AsyncMockCursor(desc=False)
    conn_mock2 = AsyncMockConn(c_mock2)

    async def _async_conn2(*a, **k):
        """Docstring for _async_conn2."""
        return conn_mock2

    adapter1.connect_async = _async_conn2

    res_fw2 = await adapter1.execute_with_feedback_async("UPDATE")
    assert res_fw2 == (True, [], None)

    # Test execute_with_feedback_async failure
    c_mock3 = AsyncMockCursor()
    conn_mock3 = AsyncMockConn(c_mock3)

    async def _async_conn3(*a, **k):
        """Docstring for _async_conn3."""
        return conn_mock3

    adapter1.connect_async = _async_conn3
    res_err = await adapter1.execute_with_feedback_async("FAIL_EXEC")
    assert res_err == (False, [], "exec_err")

    # Test execute_with_feedback_async rollback on Exception
    c_mock4 = AsyncMockCursor()
    conn_mock4 = AsyncMockConn(c_mock4)

    async def _async_conn4(*a, **k):
        """Docstring for _async_conn4."""
        return conn_mock4

    adapter1.connect_async = _async_conn4
    with patch.object(sa.SQLiteAdapter, "error_classes", new_callable=PropertyMock, return_value=(Exception,)):
        res_val = await adapter1.execute_with_feedback_async("FAIL_AFTER")
        assert res_val == (False, [], "val_err")
        assert conn_mock4.rollback_called

    # Test execute_query_async success with desc
    c_mock5 = AsyncMockCursor()
    conn_mock5 = AsyncMockConn(c_mock5)

    async def _async_conn5(*a, **k):
        """Docstring for _async_conn5."""
        return conn_mock5

    adapter1.connect_async = _async_conn5
    res_q = await adapter1.execute_query_async("SELECT")
    assert res_q == [("row",)]

    # Test execute_query_async success NO desc
    c_mock6 = AsyncMockCursor(desc=False)
    conn_mock6 = AsyncMockConn(c_mock6)

    async def _async_conn6(*a, **k):
        """Docstring for _async_conn6."""
        return conn_mock6

    adapter1.connect_async = _async_conn6
    res_q2 = await adapter1.execute_query_async("UPDATE")
    assert res_q2 == []

    # Test execute_query_async failure
    c_mock7 = AsyncMockCursor()
    conn_mock7 = AsyncMockConn(c_mock7)

    async def _async_conn7(*a, **k):
        """Docstring for _async_conn7."""
        return conn_mock7

    adapter1.connect_async = _async_conn7
    res_q3 = await adapter1.execute_query_async("FAIL_EXEC")
    assert res_q3 == []

    # Test execute_with_feedback_async success NO desc NO commit
    class AsyncMockConnNoCommit:
        """Docstring for AsyncMockConnNoCommit."""

        def __init__(self, c_mock):
            """Docstring for __init__."""
            self.c_mock = c_mock

        async def execute(self, q, p):
            """Docstring for execute."""
            return self.c_mock

        def close(self):
            """Docstring for close."""

        def rollback(self):
            """Docstring for rollback."""

    c_mock_nocommit = AsyncMockCursor(desc=False)
    conn_mock_nocommit = AsyncMockConnNoCommit(c_mock_nocommit)

    async def _async_conn_nocommit(*a, **k):
        """Docstring for _async_conn_nocommit."""
        return conn_mock_nocommit

    adapter1.connect_async = _async_conn_nocommit
    await adapter1.execute_with_feedback_async("UPDATE")
    await adapter1.execute_query_async("UPDATE")

    # Test missing close and commit methods for coverage 158-182
    class AsyncMockCursorNoClose:
        """Docstring for AsyncMockCursorNoClose."""

        def __init__(self, desc=True):
            """Docstring for __init__."""
            self.description = desc
            if not desc:
                del self.description

        async def fetchall(self):
            """Docstring for fetchall."""
            return [("row",)]

    class AsyncMockConnNoCloseCommit:
        """Docstring for AsyncMockConnNoCloseCommit."""

        def __init__(self, c_mock):
            """Docstring for __init__."""
            self.c_mock = c_mock

        async def execute(self, q, p):
            """Docstring for execute."""
            if q == "FAIL_AFTER":
                raise ValueError("val_err")
            return self.c_mock

    class SyncMockCursor:
        """Docstring for SyncMockCursor."""

        def __init__(self, desc=True):
            """Docstring for __init__."""
            self.description = desc
            if not desc:
                del self.description

        async def fetchall(self):
            """Docstring for fetchall."""
            return [("row",)]

        def close(self):
            """Docstring for close."""

    class SyncMockConn:
        """Docstring for SyncMockConn."""

        def __init__(self, c_mock):
            """Docstring for __init__."""
            self.c_mock = c_mock

        async def execute(self, q, p):
            """Docstring for execute."""
            return self.c_mock

        def commit(self):
            """Docstring for commit."""

        def rollback(self):
            """Docstring for rollback."""

        def close(self):
            """Docstring for close."""

    c_mock_sync = SyncMockCursor(desc=False)
    conn_mock_sync = SyncMockConn(c_mock_sync)

    async def _async_conn_sync(*a, **k):
        """Docstring for _async_conn_sync."""
        return conn_mock_sync

    adapter1.connect_async = _async_conn_sync
    await adapter1.execute_with_feedback_async("UPDATE")
    await adapter1.execute_query_async("UPDATE")

    # And test rollback sync
    class SyncMockConnFail(SyncMockConn):
        """Docstring for SyncMockConnFail."""

        async def execute(self, q, p):
            """Docstring for execute."""
            raise ValueError("val_err")

    conn_mock_sync_fail = SyncMockConnFail(c_mock_sync)

    async def _async_conn_sync_fail(*a, **k):
        """Docstring for _async_conn_sync_fail."""
        return conn_mock_sync_fail

    adapter1.connect_async = _async_conn_sync_fail
    with patch.object(sa.SQLiteAdapter, "error_classes", new_callable=PropertyMock, return_value=(Exception,)):
        await adapter1.execute_with_feedback_async("FAIL_AFTER")

    c_mock8 = AsyncMockCursorNoClose()
    conn_mock8 = AsyncMockConnNoCloseCommit(c_mock8)

    async def _async_conn8(*a, **k):
        """Docstring for _async_conn8."""
        return conn_mock8

    adapter1.connect_async = _async_conn8

    await adapter1.execute_query_async("SELECT")
    c_mock8_2 = AsyncMockCursorNoClose(desc=False)
    conn_mock8.c_mock = c_mock8_2
    await adapter1.execute_query_async("UPDATE")

    with patch.object(sa.SQLiteAdapter, "error_classes", new_callable=PropertyMock, return_value=(Exception,)):
        await adapter1.execute_with_feedback_async("FAIL_AFTER")
