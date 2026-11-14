"""Unit tests for the base `DatabaseAdapter` class using strongly-typed protocol test doubles."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping, Sequence
from typing import Self

import pytest

from gemma_4_sql.exceptions import DatabaseExecutionError
from gemma_4_sql.sdk.adapters.base import AsyncAdapterConnection, DatabaseAdapter, SyncAdapterConnection
from gemma_4_sql.type_hints import JSONPrimitive


class _DirectResult:
    """Fetchable execution result returned directly by `_DirectExecConn.execute`."""

    def __init__(self, rows: Sequence[Sequence[JSONPrimitive]]) -> None:
        """Store rows to return on `fetchall()`.

        Args:
            rows: Sequence of result rows.

        """
        self._rows = rows

    def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
        """Return stored rows.

        Returns:
            Result rows.

        """
        return self._rows


class _DirectExecConn:
    """Connection double without `.cursor()` or `.close()` supporting `.execute(...).fetchall()`."""

    def __init__(self, rows: Sequence[Sequence[JSONPrimitive]] = (("row",),)) -> None:
        """Initialize direct-execute connection double.

        Args:
            rows: Rows returned by `execute(...).fetchall()`.

        """
        self._rows = rows

    def execute(
        self,
        _query: str,
        _parameters: Sequence[JSONPrimitive] = (),
        /,
    ) -> _DirectResult:
        """Execute query and return `_DirectResult`.

        Args:
            _query: SQL query text.
            _parameters: Optional query parameters.

        Returns:
            `_DirectResult` wrapping `self._rows`.

        """
        return _DirectResult(self._rows)


class _TypedCursorNoClose:
    """Cursor double with `description` and `fetchall` but no `close` method."""

    description: Sequence[Sequence[str | int | None]] | bool | None = True

    def execute(
        self,
        _query: str,
        _parameters: Sequence[JSONPrimitive] | Mapping[str, JSONPrimitive] | None = None,
        /,
    ) -> Self:
        """Execute query no-op.

        Args:
            _query: SQL query string.
            _parameters: Optional parameters.

        Returns:
            `self`.

        """
        return self

    def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
        """Return single row.

        Returns:
            List containing `('c_row',)`.

        """
        return [("c_row",)]


class _TypedCursorWithClose:
    """Cursor double with configurable `description`, `execute` failure, and `close` failure."""

    def __init__(
        self,
        *,
        has_description: bool = True,
        exec_error: str | None = None,
        close_error: str | None = None,
    ) -> None:
        """Initialize cursor double.

        Args:
            has_description: Whether `description` is non-`None`.
            exec_error: Optional `ValueError` message raised on `execute`.
            close_error: Optional `ValueError` message raised on `close`.

        """
        self.description: Sequence[Sequence[str | int | None]] | bool | None = True if has_description else None
        self._exec_error = exec_error
        self._close_error = close_error
        self.closed = False

    def execute(
        self,
        _query: str,
        _parameters: Sequence[JSONPrimitive] | Mapping[str, JSONPrimitive] | None = None,
        /,
    ) -> Self:
        """Execute query or raise `ValueError`.

        Args:
            _query: SQL query string.
            _parameters: Optional parameters.

        Returns:
            `self`.

        Raises:
            ValueError: If `_exec_error` is configured.

        """
        if self._exec_error is not None:
            raise ValueError(self._exec_error)
        return self

    def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
        """Return single row.

        Returns:
            List containing `('c_row',)`.

        """
        return [("c_row",)]

    def close(self) -> None:
        """Close cursor or raise `ValueError`.

        Raises:
            ValueError: If `_close_error` is configured.

        """
        self.closed = True
        if self._close_error is not None:
            raise ValueError(self._close_error)


class _CursorFactoryConn:
    """Connection double exposing `.cursor()` and `.close()`."""

    def __init__(self, cursor_obj: _TypedCursorNoClose | _TypedCursorWithClose) -> None:
        """Store cursor double returned by `cursor()`.

        Args:
            cursor_obj: Cursor test double.

        """
        self._cursor_obj = cursor_obj
        self.closed = 0

    def cursor(self) -> _TypedCursorNoClose | _TypedCursorWithClose:
        """Return configured cursor double.

        Returns:
            Cursor double.

        """
        return self._cursor_obj

    def close(self) -> None:
        """Increment close counter."""
        self.closed += 1


class _AsyncCoroCloseConn(_DirectExecConn):
    """Async connection double whose `.close()` returns an awaitable coroutine."""

    def __init__(self, *, raise_on_close: bool = False) -> None:
        """Initialize async close double.

        Args:
            raise_on_close: Whether the close coroutine raises `OSError`.

        """
        super().__init__()
        self.closed = 0
        self._raise_on_close = raise_on_close

    def close(self) -> Coroutine[None, None, None]:
        """Return an async close coroutine.

        Returns:
            Coroutine incrementing `self.closed`.

        """

        async def _do_close() -> None:
            """Increment close count or raise `OSError`.

            Raises:
                OSError: When `_raise_on_close` is configured.

            """
            self.closed += 1
            if self._raise_on_close:
                raise OSError("close failed")

        return _do_close()


class DummyAdapter(DatabaseAdapter[SyncAdapterConnection, AsyncAdapterConnection]):
    """Concrete `DatabaseAdapter` subclass for unit testing base behavior."""

    connect_calls: int

    def __init__(
        self,
        db_path: str,
        db_kwargs: dict[str, str | int | SyncAdapterConnection],
        *,
        lazy_connect: bool = False,
    ) -> None:
        """Initialize `DummyAdapter` tracking `connect()` call counts.

        Args:
            db_path: Database path.
            db_kwargs: Connection options.
            lazy_connect: Whether to defer initial connection until `.conn` access.

        """
        self.connect_calls = 0
        super().__init__(db_path, dict(db_kwargs), lazy_connect=lazy_connect)

    def connect(self) -> SyncAdapterConnection:
        """Return a `_DirectExecConn` connection double.

        Returns:
            `_DirectExecConn` instance.

        """
        self.connect_calls += 1
        return _DirectExecConn()

    async def connect_async(self) -> AsyncAdapterConnection:
        """Connect asynchronously via worker thread.

        Returns:
            `DatabaseConnectionProtocol` instance.

        """
        return await asyncio.to_thread(self.connect)

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Provide handled exception classes for `DummyAdapter`.

        Returns:
            Tuple containing `ValueError`.

        """
        return (ValueError,)


@pytest.mark.asyncio
async def test_base_adapter_sync_async_lazy_and_schema_errors() -> None:
    """Verify sync/async execution, lazy connection lifecycle, close tasks, and `setup_schema` errors."""
    lazy_adapter = DummyAdapter("path", {}, lazy_connect=True)
    assert lazy_adapter.connect_calls == 0
    _ = lazy_adapter.conn
    assert lazy_adapter.connect_calls == 1

    adapter = DummyAdapter("path", {})
    assert isinstance(await adapter.connect_async(), _DirectExecConn)

    assert adapter.execute_with_feedback("SELECT") == (True, [("row",)], None)
    assert await adapter.execute_with_feedback_async("SELECT") == (True, [("row",)], None)
    assert adapter.execute_query("SELECT") == [("row",)]
    assert await adapter.execute_query_async("SELECT") == [("row",)]

    # Closing connection without `.close()` attribute is a safe no-op
    adapter.close()

    # Cursor without `.close()` method
    conn_no_close_cur = _CursorFactoryConn(_TypedCursorNoClose())
    adapter.conn = conn_no_close_cur
    assert adapter.execute_with_feedback("SELECT") == (True, [("c_row",)], None)
    assert adapter.execute_query("SELECT") == [("c_row",)]
    adapter.setup_schema("CREATE TABLE t (x INT)")
    adapter.close()
    assert conn_no_close_cur.closed == 1

    # Cursor with `description=None` returns empty list and closes cursor
    cur_no_desc = _TypedCursorWithClose(has_description=False)
    adapter.conn = _CursorFactoryConn(cur_no_desc)
    assert adapter.execute_query("INSERT INTO t VALUES (1)") == []
    assert cur_no_desc.closed is True

    # Execution failure raises DatabaseExecutionError in both execute_query and setup_schema!
    adapter.conn = _CursorFactoryConn(_TypedCursorWithClose(exec_error="bad ddl"))
    assert adapter.execute_with_feedback("SELECT") == (False, [], "bad ddl")
    with pytest.raises(DatabaseExecutionError, match="bad ddl"):
        adapter.execute_query("SELECT")
    with pytest.raises(DatabaseExecutionError, match="bad ddl"):
        adapter.setup_schema("BAD DDL")

    # Cursor close failure also propagates as DatabaseExecutionError
    adapter.conn = _CursorFactoryConn(_TypedCursorWithClose(close_error="err_close"))
    with pytest.raises(DatabaseExecutionError, match="err_close"):
        adapter.execute_query("SELECT")

    # Cursor with close method succeeds cleanly
    cur_ok = _TypedCursorWithClose()
    adapter.conn = _CursorFactoryConn(cur_ok)
    assert adapter.execute_with_feedback("SELECT") == (True, [("c_row",)], None)
    assert cur_ok.closed is True

    # Schedule async close while event loop is running, then await pending tasks via close_async()
    coro_conn = _AsyncCoroCloseConn()
    adapter._async_conn = coro_conn
    adapter.close()
    await adapter.close_async()
    assert coro_conn.closed == 1


def test_base_adapter_sync_close_outside_running_loop() -> None:
    """Verify `close()` runs coroutine `.close()` outside an event loop and tolerates `OSError`."""
    adapter = DummyAdapter("path", {})
    coro_ok = _AsyncCoroCloseConn(raise_on_close=False)
    adapter._async_conn = coro_ok
    adapter.close()
    assert coro_ok.closed == 1

    coro_err = _AsyncCoroCloseConn(raise_on_close=True)
    adapter._async_conn = coro_err
    adapter.close()
    assert coro_err.closed == 1

    sync_cached = _CursorFactoryConn(_TypedCursorNoClose())
    adapter._async_conn = sync_cached
    adapter.close()
    assert sync_cached.closed == 1
