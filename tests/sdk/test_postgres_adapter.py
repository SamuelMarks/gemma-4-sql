"""Unit tests for the PostgreSQL database adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.sdk.adapters.postgres_adapter as pa
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from tests.utils.test_utils import FakeDBConnection


@pytest.fixture
def fake_postgres_module() -> tuple[MagicMock, MagicMock, FakeDBConnection, type[Exception]]:
    """Create mocked psycopg2 and asyncpg modules backed by FakeDBConnection.

    Returns:
        Tuple of `(mock_psycopg2, mock_asyncpg, fake_conn, PostgresError)`.

    """
    mock_psycopg2 = MagicMock()

    class PostgresError(Exception):
        """Simulated PostgreSQL driver error."""

    mock_psycopg2.Error = PostgresError

    conn = FakeDBConnection()
    mock_psycopg2.connect.return_value = conn

    mock_asyncpg = MagicMock()
    mock_asyncpg.PostgresError = PostgresError

    async def mock_ac(*_a: Any, **_k: Any) -> Any:
        """Create an async connection wrapper around FakeDBConnection."""

        class AsyncFakeConn:
            """Async wrapper over FakeDBConnection."""

            def __init__(self) -> None:
                """Initialize with shared FakeDBConnection."""
                self._conn = conn

            async def fetch(self, query: str, *args: Any) -> list[Any]:
                """Execute query on FakeDBConnection and return MockRecord instances."""
                cur = self._conn.cursor()
                cur.execute(query, args)

                class MockRecord:
                    """Simulated asyncpg.Record."""

                    def __init__(self, row: tuple[object, ...]) -> None:
                        """Store row tuple."""
                        self._row = row

                    def values(self) -> list[Any]:
                        """Return row values as list."""
                        return list(self._row)

                return [MockRecord(row) for row in cur.fetchall()]

            async def close(self) -> None:
                """Close async connection no-op."""

        return AsyncFakeConn()

    mock_asyncpg.connect.side_effect = mock_ac

    return mock_psycopg2, mock_asyncpg, conn, PostgresError


@pytest.mark.asyncio
async def test_postgres_adapter_execution(
    fake_postgres_module: tuple[MagicMock, MagicMock, FakeDBConnection, type[Exception]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PostgresAdapter synchronous and asynchronous execution and error propagation."""
    mock_psycopg2, mock_asyncpg, conn, postgres_error_cls = fake_postgres_module

    monkeypatch.setattr(pa, "psycopg2", mock_psycopg2)
    monkeypatch.setattr(pa, "asyncpg", mock_asyncpg)

    adapter = pa.PostgresAdapter("host=localhost dbname=test", {})

    empty_adapter = pa.PostgresAdapter("", {})
    empty_adapter.connect()
    await empty_adapter.connect_async()

    conn.set_results([[(1, "apple")]])
    res = await adapter.execute_query_async("SELECT")
    assert res == [(1, "apple")]

    conn.set_results([[(1, "apple")]])
    suc, res, err = await adapter.execute_with_feedback_async("SELECT")
    assert suc is True
    assert res == [(1, "apple")]
    assert err is None

    def mock_fail(*_args: Any, **_kwargs: Any) -> None:
        """Raise simulated PostgresError."""
        msg = "Simulated"
        raise postgres_error_cls(msg)

    conn.cursor = MagicMock()  # type: ignore[method-assign]
    conn.cursor.return_value.execute = mock_fail

    suc, res, err = await adapter.execute_with_feedback_async("FAIL")
    assert suc is False
    assert res == []
    assert "Simulated" in str(err)

    with pytest.raises(DatabaseExecutionError, match="Simulated"):
        await adapter.execute_query_async("FAIL")

    await adapter.close_async()
    await adapter.close_async()
    await empty_adapter.close_async()
    adapter.close()
    empty_adapter.close()


@pytest.mark.asyncio
async def test_postgres_adapter_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify PostgresAdapter raises DependencyMissingError when psycopg2 or asyncpg is missing."""
    monkeypatch.setattr(pa, "psycopg2", None)
    with pytest.raises(DependencyMissingError, match="psycopg2 is required"):
        pa.PostgresAdapter("path", {})

    monkeypatch.setattr(pa, "psycopg2", MagicMock())
    monkeypatch.setattr(pa, "asyncpg", None)

    adapter = pa.PostgresAdapter("path", {})
    with pytest.raises(DependencyMissingError, match="asyncpg is required"):
        await adapter.connect_async()

    monkeypatch.setattr(pa, "psycopg2", None)
    monkeypatch.setattr(pa, "asyncpg", None)
    with patch.object(pa.PostgresAdapter, "__init__", return_value=None):
        dummy_adapter = pa.PostgresAdapter("path", {})
        assert Exception in dummy_adapter.error_classes

        mock_p = MagicMock()
        mock_p.Error = 123
        monkeypatch.setattr(pa, "psycopg2", mock_p)

        mock_a = MagicMock()
        mock_a.PostgresError = 123
        monkeypatch.setattr(pa, "asyncpg", mock_a)

        assert Exception in dummy_adapter.error_classes


def test_postgres_adapter_schema(
    fake_postgres_module: tuple[MagicMock, MagicMock, FakeDBConnection, type[Exception]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PostgresAdapter.setup_schema commits DDL and propagates execution failures."""
    mock_psycopg2, mock_asyncpg, conn, _ = fake_postgres_module

    monkeypatch.setattr(pa, "psycopg2", mock_psycopg2)
    monkeypatch.setattr(pa, "asyncpg", mock_asyncpg)

    adapter = pa.PostgresAdapter("path", {})
    adapter.setup_schema("CREATE TABLE t (a INT)")
    assert conn._transactions_committed == 1

    conn.set_fail_query("FAIL")
    with pytest.raises(Exception, match="Simulated execution failure"):
        adapter.setup_schema("FAIL")
