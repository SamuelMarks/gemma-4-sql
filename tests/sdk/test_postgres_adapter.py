"""Unit tests for `PostgresAdapter` using strongly-typed protocol test doubles."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Self

import pytest

import gemma_4_sql.sdk.adapters.postgres_adapter as pa
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.sdk.adapters.base import DBKwargValue, MinimalCursorProtocol
from gemma_4_sql.sdk.adapters.postgres_adapter import (
    normalize_to_asyncpg_placeholders,
    normalize_to_psycopg2_placeholders,
)
from gemma_4_sql.type_hints import DatabaseConnectionProtocol, JSONPrimitive
from tests.utils.test_utils import FakeDBConnection


class _PostgresError(Exception):
    """Simulated PostgreSQL driver error."""


class _MockPgRecord:
    """Typed `asyncpg.Record` test double."""

    def __init__(self, row: tuple[JSONPrimitive, ...]) -> None:
        """Store row tuple.

        Args:
            row: Tuple of column values.

        """
        self._row = row

    def values(self) -> list[JSONPrimitive]:
        """Return row values.

        Returns:
            List of column values.

        """
        return list(self._row)


class _AsyncFakePgConn:
    """Typed async connection test double wrapping `FakeDBConnection`."""

    def __init__(
        self,
        conn: FakeDBConnection,
        *,
        fail_readonly_set: bool = False,
        fail_close: bool = False,
    ) -> None:
        """Initialize async PostgreSQL connection double.

        Args:
            conn: Shared `FakeDBConnection`.
            fail_readonly_set: Whether `SET DEFAULT_TRANSACTION_READ_ONLY` raises `_PostgresError`.
            fail_close: Whether `close()` raises `OSError`.

        """
        self._conn = conn
        self._fail_readonly_set = fail_readonly_set
        self._fail_close = fail_close
        self.closed = 0
        self.executed_commands: list[str] = []
        self.last_fetch: tuple[str, tuple[JSONPrimitive, ...]] | None = None

    async def execute(self, query: str) -> str:
        """Record session-level command or raise `_PostgresError`.

        Args:
            query: Session SQL statement.

        Returns:
            Status tag `'SET'`.

        Raises:
            _PostgresError: When `self._fail_readonly_set` is enabled.

        """
        if self._fail_readonly_set:
            raise _PostgresError("cannot set read-only")
        self.executed_commands.append(query)
        return "SET"

    async def fetch(self, query: str, *args: JSONPrimitive) -> list[_MockPgRecord]:
        """Execute query against `FakeDBConnection` and wrap rows in `_MockPgRecord`.

        Args:
            query: SQL query text.
            *args: Positional query arguments.

        Returns:
            List of `_MockPgRecord` rows.

        """
        self.last_fetch = (query, args)
        cur = self._conn.cursor()
        cur.execute(query, args)
        return [_MockPgRecord(row) for row in cur.fetchall()]

    async def close(self) -> None:
        """Close connection or raise `OSError` when `fail_close` is set.

        Raises:
            OSError: When `self._fail_close` is enabled.

        """
        self.closed += 1
        if self._fail_close:
            raise OSError("socket already reset")


class _FakePsycopg2Module:
    """Typed `psycopg2` module double."""

    Error: type[Exception] = _PostgresError

    def __init__(self, conn: FakeDBConnection) -> None:
        """Store `FakeDBConnection` returned on `connect(...)`.

        Args:
            conn: Underlying fake connection.

        """
        self._conn = conn

    def connect(self, dsn: str = "", **kwargs: DBKwargValue) -> DatabaseConnectionProtocol:
        """Return `self._conn`.

        Args:
            dsn: Connection string.
            **kwargs: Extra connection parameters.

        Returns:
            `FakeDBConnection` instance.

        """
        _ = (dsn, kwargs)
        return self._conn


class _FakeAsyncpgModule:
    """Typed `asyncpg` module double."""

    PostgresError: type[Exception] = _PostgresError

    def __init__(self, factory: Callable[[], _AsyncFakePgConn]) -> None:
        """Store async connection factory.

        Args:
            factory: Callable producing `_AsyncFakePgConn` instances.

        """
        self._factory = factory

    async def connect(self, dsn: str = "", **kwargs: DBKwargValue) -> _AsyncFakePgConn:
        """Return a new `_AsyncFakePgConn`.

        Args:
            dsn: Connection string.
            **kwargs: Extra connection parameters.

        Returns:
            `_AsyncFakePgConn` instance.

        """
        _ = (dsn, kwargs)
        return self._factory()


@pytest.fixture
def fake_postgres_bundle() -> tuple[_FakePsycopg2Module, _FakeAsyncpgModule, FakeDBConnection, list[_AsyncFakePgConn]]:
    """Provide typed `psycopg2` and `asyncpg` module doubles plus created async connections.

    Returns:
        Tuple of `(psycopg2_mod, asyncpg_mod, sync_conn, created_async_conns)`.

    """
    conn = FakeDBConnection()
    created_async: list[_AsyncFakePgConn] = []

    def _make_async() -> _AsyncFakePgConn:
        """Create and record an `_AsyncFakePgConn`.

        Returns:
            Newly created `_AsyncFakePgConn`.

        """
        ac = _AsyncFakePgConn(conn)
        created_async.append(ac)
        return ac

    return (_FakePsycopg2Module(conn), _FakeAsyncpgModule(_make_async), conn, created_async)


def test_postgres_placeholder_normalization() -> None:
    """Verify `%s` <-> `$1` normalization rewrites unquoted placeholders while preserving literals and comments."""
    q_pct = "SELECT '%s' AS lit, /* %s */ name FROM users WHERE id = %s AND role = %s"
    norm_async, p_async = normalize_to_asyncpg_placeholders(q_pct, [42, "admin"])
    assert norm_async == "SELECT '%s' AS lit, /* %s */ name FROM users WHERE id = $1 AND role = $2"
    assert p_async == (42, "admin")
    assert normalize_to_asyncpg_placeholders("SELECT 1", []) == ("SELECT 1", ())

    q_dlr = "SELECT '$1' AS lit, /* $2 */ name FROM users WHERE role = $2 AND id = $1 AND other = $9"
    norm_sync, p_sync = normalize_to_psycopg2_placeholders(q_dlr, [42, "admin"])
    assert norm_sync == "SELECT '$1' AS lit, /* $2 */ name FROM users WHERE role = %s AND id = %s AND other = $9"
    assert p_sync == ("admin", 42)
    assert normalize_to_psycopg2_placeholders("SELECT * FROM users WHERE id = %s", [7]) == (
        "SELECT * FROM users WHERE id = %s",
        (7,),
    )
    assert normalize_to_psycopg2_placeholders("SELECT 1", []) == ("SELECT 1", ())


@pytest.mark.asyncio
async def test_postgres_adapter_execution_and_socket_cleanup(
    fake_postgres_bundle: tuple[_FakePsycopg2Module, _FakeAsyncpgModule, FakeDBConnection, list[_AsyncFakePgConn]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify sync/async execution, placeholder normalization, read-only session setup, and socket cleanup on error."""
    pg_mod, apg_mod, conn, created_async = fake_postgres_bundle

    adapter = pa.PostgresAdapter(
        "host=localhost dbname=test",
        {},
        psycopg2_module=pg_mod,
        asyncpg_module=apg_mod,
    )
    empty_adapter = pa.PostgresAdapter(
        "",
        {},
        read_only=False,
        psycopg2_module=pg_mod,
        asyncpg_module=apg_mod,
    )
    empty_adapter.connect()
    await empty_adapter.connect_async()

    conn.set_results([[(1, "apple")]])
    res_sync = adapter.execute_query("SELECT * FROM items WHERE id = $1", [1])
    assert res_sync == [(1, "apple")]

    conn.set_results([[(1, "apple")]])
    res = await adapter.execute_query_async("SELECT * FROM items WHERE id = %s", [1])
    assert res == [(1, "apple")]
    assert created_async[0].executed_commands == []
    assert created_async[1].executed_commands == ["SET DEFAULT_TRANSACTION_READ_ONLY = ON"]
    assert created_async[1].last_fetch == ("SELECT * FROM items WHERE id = $1", (1,))
    # Cached async connection is reused on second query
    conn.set_results([[(1, "apple")]])
    suc, rows, err = await adapter.execute_with_feedback_async("SELECT")
    assert suc is True and rows == [(1, "apple")] and err is None

    # Error during async query closes the failed `_async_conn` instead of leaking the socket!
    class _FailingCursor:
        """Cursor double raising `_PostgresError`."""

        description: Sequence[Sequence[str | int | None]] | bool | None = None

        def execute(
            self,
            _q: str,
            _p: Sequence[JSONPrimitive] | Mapping[str, JSONPrimitive] | None = None,
            /,
        ) -> Self:
            """Raise `_PostgresError`.

            Raises:
                _PostgresError: Always raised to simulate query failure.

            """
            raise _PostgresError("Simulated")

        def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
            """Return empty rows."""
            return []

        def close(self) -> None:
            """No-op close."""

    assert _FailingCursor().fetchall() == []
    monkeypatch.setattr(conn, "cursor", _FailingCursor)

    first_async_conn = created_async[1]
    suc_err, rows_err, err_str = await adapter.execute_with_feedback_async("FAIL")
    assert suc_err is False and rows_err == [] and "Simulated" in str(err_str)
    assert first_async_conn.closed == 1

    # Sync close() returning None on failed async connection + connect_async raising
    # _PostgresError when _async_conn is None
    class _SyncCloseFailPgConn:
        """Async connection double whose `.close()` returns `None` synchronously."""

        def __init__(self) -> None:
            """Initialize close counter."""
            self.closed = 0

        async def fetch(self, _query: str, *_args: JSONPrimitive) -> list[_MockPgRecord]:
            """Raise `_PostgresError`.

            Raises:
                _PostgresError: Always raised to simulate fetch failure.

            """
            raise _PostgresError("Simulated")

        def close(self) -> None:
            """Increment closed count synchronously."""
            self.closed += 1

    sync_fail_conn = _SyncCloseFailPgConn()

    class _SyncFailAsyncpgModule:
        """Asyncpg module double returning `_SyncCloseFailPgConn`."""

        __name__: str = "asyncpg"
        PostgresError: type[Exception] = _PostgresError

        async def connect(self, dsn: str = "", /, **kwargs: DBKwargValue) -> _SyncCloseFailPgConn:
            """Return `sync_fail_conn`."""
            _ = (dsn, kwargs)
            return sync_fail_conn

    sync_fail_adapter = pa.PostgresAdapter(
        "dsn",
        {},
        lazy_connect=True,
        psycopg2_module=pg_mod,
        asyncpg_module=_SyncFailAsyncpgModule(),
    )
    with pytest.raises(DatabaseExecutionError, match="Simulated"):
        await sync_fail_adapter.execute_query_async("FAIL")
    assert sync_fail_conn.closed == 1

    def _raise_pg_connect() -> _AsyncFakePgConn:
        """Raise `_PostgresError` before `_async_conn` is set.

        Raises:
            _PostgresError: Always raised to simulate connection error.

        """
        raise _PostgresError("connect boom")

    conn_err_adapter = pa.PostgresAdapter(
        "dsn",
        {},
        lazy_connect=True,
        psycopg2_module=pg_mod,
        asyncpg_module=_FakeAsyncpgModule(_raise_pg_connect),
    )
    with pytest.raises(DatabaseExecutionError, match="connect boom"):
        await conn_err_adapter.execute_query_async("FAIL")

    # Also test when failed_conn.close() itself raises OSError + readonly SET raises _PostgresError
    fail_close_conn = _AsyncFakePgConn(conn, fail_readonly_set=True, fail_close=True)
    fail_close_adapter = pa.PostgresAdapter(
        "dsn",
        {},
        lazy_connect=True,
        psycopg2_module=pg_mod,
        asyncpg_module=_FakeAsyncpgModule(lambda: fail_close_conn),
    )
    with pytest.raises(DatabaseExecutionError, match="Simulated"):
        await fail_close_adapter.execute_query_async("FAIL")
    assert fail_close_conn.closed == 1

    await adapter.close_async()
    await empty_adapter.close_async()


@pytest.mark.asyncio
async def test_postgres_adapter_missing_deps_and_lazy_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `PostgresAdapter` handles missing drivers, lazy resolution, and non-exception `Error` attributes."""
    monkeypatch.setattr(pa, "psycopg2", pa._LoaderSentinel.UNRESOLVED)
    monkeypatch.setattr(pa, "asyncpg", pa._LoaderSentinel.UNRESOLVED)
    _ = pa._resolve_psycopg2()
    _ = pa._resolve_asyncpg()

    monkeypatch.setattr(pa, "psycopg2", None)
    with pytest.raises(DependencyMissingError, match="psycopg2 is required"):
        pa.PostgresAdapter("path", {})

    # With lazy_connect=True, async-only usage does NOT require psycopg2!
    lazy_pg = pa.PostgresAdapter("path", {}, lazy_connect=True)
    monkeypatch.setattr(pa, "asyncpg", None)
    with pytest.raises(DependencyMissingError, match="asyncpg is required"):
        await lazy_pg.connect_async()
    assert DatabaseExecutionError in lazy_pg.error_classes

    class _InvalidErrorModule:
        """Module stub with non-type `Error` and `PostgresError` attributes."""

        Error = 123
        PostgresError = 456

    monkeypatch.setattr(pa, "psycopg2", _InvalidErrorModule())
    monkeypatch.setattr(pa, "asyncpg", _InvalidErrorModule())
    assert DatabaseExecutionError in lazy_pg.error_classes


def test_postgres_adapter_schema(
    fake_postgres_bundle: tuple[_FakePsycopg2Module, _FakeAsyncpgModule, FakeDBConnection, list[_AsyncFakePgConn]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `PostgresAdapter.setup_schema` commits on success and raises `DatabaseExecutionError` on failure."""
    pg_mod, apg_mod, conn, _ = fake_postgres_bundle
    monkeypatch.setattr(pa, "psycopg2", pg_mod)
    monkeypatch.setattr(pa, "asyncpg", apg_mod)

    adapter = pa.PostgresAdapter("path", {})
    adapter.setup_schema("CREATE TABLE t (a INT)")
    assert conn._transactions_committed == 1

    conn.set_fail_query("FAIL")
    with pytest.raises(DatabaseExecutionError, match="Simulated execution failure"):
        adapter.setup_schema("FAIL")
    assert conn._transactions_rolled_back == 1

    class StatelessPgConn:
        """PostgreSQL connection stub without commit/rollback methods."""

        def __init__(self, *, should_fail: bool = False) -> None:
            """Initialize with optional failure flag."""
            self.should_fail = should_fail

        def execute(
            self,
            _q: str,
            _p: Sequence[JSONPrimitive] | Mapping[str, JSONPrimitive] | None = None,
            /,
        ) -> MinimalCursorProtocol:
            """Delegate to cursor.

            Returns:
                Minimal cursor instance.

            """
            return self.cursor()

        def cursor(self) -> MinimalCursorProtocol:
            """Return minimal cursor without `.close()`."""
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
                    """Execute DDL or raise `RuntimeError`.

                    Returns:
                        `self`.

                    Raises:
                        RuntimeError: When `outer.should_fail` is enabled.

                    """
                    if outer.should_fail:
                        raise RuntimeError("Mock error")
                    return self

                def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
                    """Return empty result set."""
                    return []

            return _Cur()

        def close(self) -> None:
            """Close connection no-op."""

    pg_ok = StatelessPgConn(should_fail=False)
    assert pg_ok.execute("SELECT 1").fetchall() == []
    adapter.conn = pg_ok
    adapter.setup_schema("CREATE TABLE t2 (a INT)")

    adapter.conn = StatelessPgConn(should_fail=True)
    with pytest.raises(DatabaseExecutionError, match="Mock error"):
        adapter.setup_schema("FAIL2")

    # Sync `connect()` with `read_only=True` when SET raises `_PostgresError` and cursor has no `.close()`
    class _FailSetPgModule:
        """Psycopg2 module stub whose cursor raises `_PostgresError` on `SET`."""

        Error: type[Exception] = _PostgresError

        def connect(self, dsn: str = "", /, **kwargs: DBKwargValue) -> StatelessPgConn:
            """Return `StatelessPgConn` raising `_PostgresError`."""
            _ = (dsn, kwargs)

            class _ErrConn(StatelessPgConn):
                """Connection raising `_PostgresError` on cursor execute."""

                def cursor(self) -> MinimalCursorProtocol:
                    """Return cursor raising `_PostgresError`."""

                    class _ErrCur:
                        """Cursor raising `_PostgresError`."""

                        def execute(
                            self,
                            _q: str,
                            _params: Sequence[JSONPrimitive] | Mapping[str, JSONPrimitive] | None = None,
                            /,
                        ) -> Self:
                            """Raise `_PostgresError`.

                            Raises:
                                _PostgresError: Always raised to simulate denied SET command.

                            """
                            raise _PostgresError("set readonly denied")

                        def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
                            """Return empty rows."""
                            return []

                    err_cur = _ErrCur()
                    assert err_cur.fetchall() == []
                    return err_cur

            return _ErrConn()

    _ = pa.PostgresAdapter("dsn", {}, read_only=True, psycopg2_module=_FailSetPgModule())

    closed_outside_loop = False

    class SyncClosableAsyncConn:
        """Async connection stub for testing synchronous `close()` fallback."""

        async def fetch(self, _query: str, *_args: JSONPrimitive) -> Sequence[_MockPgRecord]:
            """Return empty row list."""
            return []

        async def close(self) -> None:
            """Mark connection as closed."""
            nonlocal closed_outside_loop
            closed_outside_loop = True

    sync_closable = SyncClosableAsyncConn()
    assert asyncio.run(sync_closable.fetch("SELECT 1")) == []

    class _SyncClosableAsyncpgModule:
        """Typed `asyncpg` module double returning `sync_closable`."""

        __name__: str = "asyncpg"
        PostgresError: type[Exception] = _PostgresError

        async def connect(self, dsn: str = "", /, **kwargs: DBKwargValue) -> SyncClosableAsyncConn:
            """Return `sync_closable`."""
            _ = (dsn, kwargs)
            return sync_closable

    sync_close_adapter = pa.PostgresAdapter(
        "dsn",
        {},
        read_only=False,
        lazy_connect=True,
        psycopg2_module=pg_mod,
        asyncpg_module=_SyncClosableAsyncpgModule(),
    )
    assert asyncio.run(sync_close_adapter.connect_async()) is sync_closable
    sync_close_adapter.close()
    assert closed_outside_loop is True
