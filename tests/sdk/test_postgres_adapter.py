"""Unit tests for the PostgreSQL database adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.sdk.adapters.postgres_adapter as pa
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.type_hints import CursorProtocol, JSONPrimitive, JSONValue
from tests.utils.test_utils import FakeDBConnection


class _MockPgRecord:
    """Simulated asyncpg.Record."""

    def __init__(self, row: tuple[JSONPrimitive, ...]) -> None:
        """Store row tuple."""
        self._row = row

    def values(self) -> list[JSONPrimitive]:
        """Return row values as list."""
        return list(self._row)


class _AsyncFakePgConn:
    """Async wrapper over FakeDBConnection."""

    def __init__(self, conn: FakeDBConnection) -> None:
        """Initialize with shared FakeDBConnection."""
        self._conn = conn

    async def fetch(self, query: str, *args: JSONPrimitive) -> list[_MockPgRecord]:
        """Execute query on FakeDBConnection and return _MockPgRecord instances."""
        cur = self._conn.cursor()
        cur.execute(query, args)
        return [_MockPgRecord(row) for row in cur.fetchall()]

    async def close(self) -> None:
        """Close async connection no-op."""


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

    async def mock_ac(*_a: str, **_k: JSONValue) -> _AsyncFakePgConn:
        """Create an async connection wrapper around FakeDBConnection."""
        return _AsyncFakePgConn(conn)

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

    def mock_fail(*_args: str | Sequence[JSONPrimitive], **_kwargs: JSONValue) -> None:
        """Raise simulated PostgresError."""
        msg = "Simulated"
        raise postgres_error_cls(msg)

    mock_cur_fn = MagicMock()
    mock_cur_fn.return_value.execute = mock_fail
    monkeypatch.setattr(conn, "cursor", mock_cur_fn)

    suc, res, err = await adapter.execute_with_feedback_async("FAIL")
    assert suc is False
    assert res == []
    assert "Simulated" in str(err)

    with pytest.raises(DatabaseExecutionError, match="Simulated"):
        await adapter.execute_query_async("FAIL")

    await adapter.close_async()
    await adapter.close_async()
    await empty_adapter.close_async()
    await adapter.connect_async()
    adapter.close()
    await adapter.close_async()
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
    assert DatabaseExecutionError in adapter.error_classes

    mock_p = MagicMock()
    mock_p.Error = 123
    monkeypatch.setattr(pa, "psycopg2", mock_p)

    mock_a = MagicMock()
    mock_a.PostgresError = 123
    monkeypatch.setattr(pa, "asyncpg", mock_a)

    assert DatabaseExecutionError in adapter.error_classes


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
    assert conn._transactions_rolled_back == 1

    class StatelessPgConn:
        """PostgreSQL connection stub without commit/rollback methods."""

        def __init__(self, should_fail: bool = False) -> None:
            """Initialize with optional failure flag."""
            self.should_fail = should_fail

        def cursor(self) -> CursorProtocol:
            """Return a minimal cursor."""
            outer = self

            class _Cur:
                """Minimal cursor stub for PostgreSQL DDL execution."""

                description: Sequence[Sequence[str | int | None]] | bool | None = None

                def execute(
                    self,
                    _q: str,
                    _params: Sequence[JSONPrimitive] | Mapping[str, JSONPrimitive] | None = None,
                    /,
                ) -> Self:
                    """Execute DDL or raise RuntimeError."""
                    if outer.should_fail:
                        raise RuntimeError("Mock error")
                    return self

                def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
                    """Return empty result set."""
                    return []

                def close(self) -> None:
                    """Close cursor no-op."""

            return _Cur()

        def close(self) -> None:
            """Close connection no-op."""

    monkeypatch.setattr(adapter, "conn", StatelessPgConn(should_fail=False))
    adapter.setup_schema("CREATE TABLE t2 (a INT)")

    monkeypatch.setattr(adapter, "conn", StatelessPgConn(should_fail=True))
    with pytest.raises(RuntimeError, match="Mock error"):
        adapter.setup_schema("FAIL2")

    closed_outside_loop = False

    class SyncClosableAsyncConn:
        """Async connection stub for testing synchronous close() fallback."""

        async def fetch(self, _query: str, *_args: JSONPrimitive) -> Sequence[Mapping[str, JSONPrimitive]]:
            """Return empty row list."""
            return []

        async def close(self) -> None:
            """Mark connection as closed."""
            nonlocal closed_outside_loop
            closed_outside_loop = True

    monkeypatch.setattr(adapter, "_async_conn", SyncClosableAsyncConn())
    adapter.close()
    assert closed_outside_loop is True
