"""Tests for PostgreSQL adapter."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from gemma_4_sql.sdk.adapters.postgres_adapter import PostgresAdapter
from tests.utils.test_utils import FakeDBConnection, FakeDBCursor


@pytest.fixture
def fake_postgres_module():
    mock_psycopg2 = MagicMock()

    class PostgresError(Exception):
        pass

    mock_psycopg2.Error = PostgresError

    conn = FakeDBConnection()
    mock_psycopg2.connect.return_value = conn

    mock_asyncpg = MagicMock()
    mock_asyncpg.PostgresError = PostgresError

    # Return fake connection
    async def mock_ac(*a, **k):
        # Create an async interface on top of FakeDBConnection
        class AsyncFakeConn:
            def __init__(self):
                self._conn = conn

            async def fetch(self, query, *args):
                cur = self._conn.cursor()
                cur.execute(query, args)

                class MockRecord:
                    def __init__(self, row):
                        self._row = row

                    def values(self):
                        return list(self._row)

                return [MockRecord(row) for row in cur.fetchall()]

            async def execute(self, query, *args):
                cur = self._conn.cursor()
                cur.execute(query, args)
                return f"CMD {cur.rowcount}"

            async def close(self):
                pass

        return AsyncFakeConn()

    mock_asyncpg.connect.side_effect = mock_ac

    return mock_psycopg2, mock_asyncpg, conn, PostgresError


@pytest.mark.asyncio
async def test_postgres_adapter_execution(fake_postgres_module, monkeypatch) -> None:
    """Test PostgresAdapter execution paths."""
    mock_psycopg2, mock_asyncpg, conn, PostgresError = fake_postgres_module
    import gemma_4_sql.sdk.adapters.postgres_adapter as pa

    monkeypatch.setattr(pa, "psycopg2", mock_psycopg2)
    monkeypatch.setattr(pa, "asyncpg", mock_asyncpg)

    adapter = pa.PostgresAdapter("host=localhost dbname=test", {})
    
    # Test empty db_path
    empty_adapter = pa.PostgresAdapter("", {})
    empty_adapter.connect()
    await empty_adapter.connect_async()

    conn.set_results([[(1, "apple")]])
    res = await adapter.execute_query_async("SELECT")
    assert res == [(1, "apple")]

    conn.set_results([[(1, "apple")]])
    print("CONN CLOSED?", conn._closed)
    suc, res, err = await adapter.execute_with_feedback_async("SELECT")
    assert suc is True
    assert res == [(1, "apple")]

    conn.set_fail_query("FAIL")

    # Make FakeDBConnection raise PostgresError on failure
    def mock_fail(*args, **kwargs):
        raise PostgresError("Simulated")

    conn.cursor = MagicMock()
    conn.cursor.return_value.execute = mock_fail

    suc, res, err = await adapter.execute_with_feedback_async("FAIL")
    assert suc is False
    assert "Simulated" in str(err)

    res_err = await adapter.execute_query_async("FAIL")
    assert res_err == []


@pytest.mark.asyncio
async def test_postgres_adapter_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test missing dependencies."""
    import gemma_4_sql.sdk.adapters.postgres_adapter as pa

    monkeypatch.setattr(pa, "psycopg2", None)
    with pytest.raises(ImportError):
        pa.PostgresAdapter("path", {})

    # Mock psycopg2 to pass init, but asyncpg is None
    monkeypatch.setattr(pa, "psycopg2", MagicMock())
    monkeypatch.setattr(pa, "asyncpg", None)
    
    adapter = pa.PostgresAdapter("path", {})
    with pytest.raises(ImportError):
        await adapter.connect_async()

    # Test error_classes branch where both are None
    monkeypatch.setattr(pa, "psycopg2", None)
    monkeypatch.setattr(pa, "asyncpg", None)
    # create adapter without init logic triggering error
    with patch.object(pa.PostgresAdapter, "__init__", return_value=None):
        dummy_adapter = pa.PostgresAdapter("path", {})
        assert Exception in dummy_adapter.error_classes

        # Test error_classes branch where Error objects are not types
        mock_p = MagicMock()
        mock_p.Error = 123
        monkeypatch.setattr(pa, "psycopg2", mock_p)
        
        mock_a = MagicMock()
        mock_a.PostgresError = 123
        monkeypatch.setattr(pa, "asyncpg", mock_a)
        
        assert Exception in dummy_adapter.error_classes


def test_postgres_adapter_schema(fake_postgres_module, monkeypatch) -> None:
    """Test setup_schema."""
    mock_psycopg2, mock_asyncpg, conn, PostgresError = fake_postgres_module
    import gemma_4_sql.sdk.adapters.postgres_adapter as pa

    monkeypatch.setattr(pa, "psycopg2", mock_psycopg2)
    monkeypatch.setattr(pa, "asyncpg", mock_asyncpg)

    adapter = pa.PostgresAdapter("path", {})
    adapter.setup_schema("CREATE TABLE t (a INT)")
    assert conn._transactions_committed == 1

    conn.set_fail_query("FAIL")
    with pytest.raises(Exception):
        adapter.setup_schema("FAIL")
