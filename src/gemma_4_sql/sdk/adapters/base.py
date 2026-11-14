"""Base adapter protocol and connection lifecycle management for database dialects."""

from __future__ import annotations

import abc
import asyncio
import logging
from collections.abc import Coroutine, Mapping, Sequence
from typing import Protocol, Self, runtime_checkable

from gemma_4_sql.exceptions import DatabaseExecutionError
from gemma_4_sql.type_hints import (
    AsyncDatabaseConnectionProtocol,
    CursorProtocol,
    DatabaseConnectionProtocol,
    JSONPrimitive,
    JSONValue,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class FetchableCursorProtocol(Protocol):
    """Protocol for fetchable database cursors passed via `db_kwargs`."""

    def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
        """Fetch all rows from the cursor.

        Returns:
            Sequence of result row sequences.

        """
        ...


@runtime_checkable
class MinimalCursorProtocol(Protocol):
    """Protocol for minimal DB-API cursors exposing `execute` and `fetchall`."""

    def execute(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | Mapping[str, JSONPrimitive] | None = ...,
        /,
    ) -> Self:
        """Execute a SQL query on the cursor.

        Args:
            query: SQL statement text.
            params: Optional bind parameters.

        Returns:
            `self` cursor instance.

        """
        ...

    def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
        """Fetch all rows from the cursor.

        Returns:
            Sequence of result rows.

        """
        ...


@runtime_checkable
class CursorFactoryConnectionProtocol(Protocol):
    """Protocol for DB-API connections exposing `.cursor()`."""

    def cursor(self) -> CursorProtocol | MinimalCursorProtocol:
        """Create and return a new database cursor.

        Returns:
            Database cursor instance.

        """
        ...


@runtime_checkable
class DirectExecConnectionProtocol(Protocol):
    """Protocol for connections supporting direct `.execute(...).fetchall()` without `.cursor()`."""

    def execute(
        self,
        query: str,
        params: Sequence[JSONPrimitive] = (),
        /,
    ) -> FetchableCursorProtocol:
        """Execute a SQL statement directly and return a fetchable result.

        Args:
            query: SQL statement text.
            params: Positional bind parameters.

        Returns:
            Fetchable result handle.

        """
        ...


@runtime_checkable
class _DescribableCursorProtocol(Protocol):
    """Protocol for cursors exposing a `.description` metadata property."""

    @property
    def description(self) -> Sequence[Sequence[str | int | None]] | bool | None:
        """Provide result column metadata (`None` for non-returning statements)."""
        ...


@runtime_checkable
class ClosableResourceProtocol(Protocol):
    """Protocol for cursors or connections exposing a synchronous or coroutine `.close()` method."""

    def close(self) -> Coroutine[None, None, None] | None:
        """Close the underlying resource.

        Returns:
            Optional awaitable coroutine when closing asynchronously, or `None`.

        """
        ...


type SyncAdapterConnection = DatabaseConnectionProtocol | CursorFactoryConnectionProtocol | DirectExecConnectionProtocol
type AsyncAdapterConnection = AsyncDatabaseConnectionProtocol | SyncAdapterConnection
type DBKwargValue = (
    JSONValue | SyncAdapterConnection | AsyncDatabaseConnectionProtocol | CursorProtocol | FetchableCursorProtocol
)


class DatabaseAdapter[
    SyncConnT: SyncAdapterConnection,
    AsyncConnT: AsyncAdapterConnection,
](abc.ABC):
    """Generic abstract base class for database dialect adapters."""

    db_path: str
    db_kwargs: dict[str, DBKwargValue]
    read_only: bool
    _conn: SyncConnT | None
    _async_conn: AsyncConnT | None
    _async_lock: asyncio.Lock
    _closing_tasks: set[asyncio.Task[None]]
    _owns_connection: bool

    def __init__(
        self,
        db_path: str,
        db_kwargs: dict[str, DBKwargValue],
        *,
        read_only: bool = False,
        lazy_connect: bool = False,
    ) -> None:
        """Initialize the database adapter and establish or defer the initial connection.

        Args:
            db_path: The file path or connection URI for the database.
            db_kwargs: Additional connection keyword arguments.
            read_only: Flag indicating whether mutating queries are disallowed.
            lazy_connect: When `True`, defer synchronous `connect()` until `.conn` is first accessed.

        """
        self.db_path = db_path
        self.db_kwargs = db_kwargs
        self.read_only = read_only
        self._conn = None
        self._async_conn = None
        self._async_lock = asyncio.Lock()
        self._closing_tasks = set()
        self._owns_connection = not (
            ("existing_conn" in db_kwargs and db_kwargs["existing_conn"] is not None)
            or ("conn" in db_kwargs and db_kwargs["conn"] is not None)
        )
        if not lazy_connect:
            self._conn = self.connect()

    @property
    def conn(self) -> SyncConnT:
        """Provide the active synchronous connection, connecting lazily if needed.

        Returns:
            Active synchronous connection instance.

        """
        if self._conn is None:
            self._conn = self.connect()
        return self._conn

    @conn.setter
    def conn(self, value: SyncConnT | None) -> None:
        """Assign or clear the cached synchronous connection instance.

        Args:
            value: Replacement synchronous connection or `None`.

        """
        self._conn = value

    def reset_connection(self) -> None:
        """Clear any cached synchronous connection reference without closing externally owned handles."""
        self._conn = None

    @property
    @abc.abstractmethod
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Provide the tuple of database-specific exception types caught by this adapter."""

    @abc.abstractmethod
    def connect(self) -> SyncConnT:
        """Connect synchronously to the underlying database.

        Returns:
            The active database connection object.

        """

    @abc.abstractmethod
    async def connect_async(self) -> AsyncConnT:
        """Connect asynchronously to the underlying database.

        Returns:
            The active asynchronous database connection object.

        """

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL statements to initialize the database schema.

        Args:
            ddl: The SQL DDL query string.

        Raises:
            DatabaseExecutionError: If DDL execution fails.

        """
        ok, _, err_msg = self.execute_with_feedback(ddl)
        if not ok:
            raise DatabaseExecutionError(err_msg or "DDL execution failed")

    def execute_with_feedback(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute a query synchronously and return execution status, rows, and error feedback.

        Args:
            query: The SQL query string to execute.
            params: Optional positional parameters for the query.

        Returns:
            A tuple containing `(success, rows, error_message)`.

        """
        try:
            rows = self.execute_query(query, params)
        except (*self.error_classes, DatabaseExecutionError) as e:
            cause = e.cause if isinstance(e, DatabaseExecutionError) and e.cause is not None else e
            return (False, [], str(cause))
        else:
            return (True, rows, None)

    async def execute_with_feedback_async(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute a query asynchronously and return execution status, rows, and error feedback.

        Args:
            query: The SQL query string to execute.
            params: Optional positional parameters for the query.

        Returns:
            A tuple containing `(success, rows, error_message)`.

        """
        try:
            rows = await self.execute_query_async(query, params)
        except (*self.error_classes, DatabaseExecutionError) as e:
            cause = e.cause if isinstance(e, DatabaseExecutionError) and e.cause is not None else e
            return (False, [], str(cause))
        else:
            return (True, rows, None)

    def execute_query(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a query synchronously and return fetched result tuples.

        Args:
            query: The SQL query string to execute.
            params: Optional positional parameters for the query.

        Returns:
            List of result row tuples.

        Raises:
            DatabaseExecutionError: If query execution fails on the underlying connection.

        """
        try:
            active_conn: SyncAdapterConnection = self.conn
            if isinstance(active_conn, CursorFactoryConnectionProtocol):
                cursor: CursorProtocol | MinimalCursorProtocol = active_conn.cursor()
                cursor.execute(query, params or ())
                desc = cursor.description if isinstance(cursor, _DescribableCursorProtocol) else None
                results: list[tuple[JSONPrimitive, ...]] = (
                    [tuple(r) for r in cursor.fetchall()] if desc is not None else []
                )
                if isinstance(cursor, ClosableResourceProtocol) and callable(cursor.close):
                    cursor.close()
                return results
            raw_rows = active_conn.execute(query, params or ()).fetchall()
            return [tuple(r) for r in raw_rows]
        except self.error_classes as e:
            logger.debug("Query execution failed: %s", e)
            raise DatabaseExecutionError(str(e), cause=e) from e

    async def execute_query_async(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a query asynchronously and return fetched result tuples.

        Args:
            query: The SQL query string to execute.
            params: Optional positional parameters for the query.

        Returns:
            List of result row tuples.

        Raises:
            DatabaseExecutionError: If query execution fails on the underlying connection.

        """
        return await asyncio.to_thread(self.execute_query, query, params)

    async def _await_closing_tasks(self) -> None:
        """Await any pending asynchronous connection close tasks."""
        if self._closing_tasks:
            pending = list(self._closing_tasks)
            self._closing_tasks.clear()
            await asyncio.gather(*pending, return_exceptions=True)

    def _schedule_or_run_async_close(self, cached_conn: AsyncAdapterConnection | None) -> None:
        """Safely close a cached asynchronous connection from a synchronous context.

        Args:
            cached_conn: Cached asynchronous connection object.

        """
        if (
            cached_conn is None
            or not isinstance(cached_conn, ClosableResourceProtocol)
            or not callable(cached_conn.close)
        ):
            return
        res = cached_conn.close()
        if asyncio.iscoroutine(res):
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(res)
                self._closing_tasks.add(task)
                task.add_done_callback(self._closing_tasks.discard)
            except RuntimeError:
                try:
                    asyncio.run(res)
                except (RuntimeError, OSError):
                    pass

    def close(self) -> None:
        """Close the active synchronous connection and safely close or schedule any cached async connection."""
        cached: AsyncAdapterConnection | None = self._async_conn
        self._async_conn = None
        self._schedule_or_run_async_close(cached)
        sync_conn: SyncAdapterConnection | None = self._conn
        self._conn = None
        if (
            sync_conn is not None
            and self._owns_connection
            and isinstance(sync_conn, ClosableResourceProtocol)
            and callable(sync_conn.close)
        ):
            sync_conn.close()

    async def close_async(self) -> None:
        """Asynchronously close any cached async connection, pending close tasks, and underlying connection."""
        await self._await_closing_tasks()
        cached: AsyncAdapterConnection | None = self._async_conn
        self._async_conn = None
        if cached is not None and isinstance(cached, ClosableResourceProtocol) and callable(cached.close):
            res = cached.close()
            if asyncio.iscoroutine(res):
                await res
        self.close()


__all__ = [
    "AsyncAdapterConnection",
    "ClosableResourceProtocol",
    "CursorFactoryConnectionProtocol",
    "DBKwargValue",
    "DatabaseAdapter",
    "DirectExecConnectionProtocol",
    "FetchableCursorProtocol",
    "MinimalCursorProtocol",
    "SyncAdapterConnection",
]
