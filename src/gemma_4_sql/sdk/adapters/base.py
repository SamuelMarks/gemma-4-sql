"""Base adapter protocol for database connections."""

from __future__ import annotations

import abc
import asyncio
import contextlib
import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

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
        """Fetch all rows from the cursor."""
        ...


type DBKwargValue = JSONValue | DatabaseConnectionProtocol | AsyncDatabaseConnectionProtocol | CursorProtocol | FetchableCursorProtocol


class DatabaseAdapter[
    SyncConnT: DatabaseConnectionProtocol,
    AsyncConnT: AsyncDatabaseConnectionProtocol | DatabaseConnectionProtocol,
](abc.ABC):
    """Generic abstract base class for database dialect adapters."""

    db_path: str
    db_kwargs: dict[str, DBKwargValue]
    read_only: bool
    _async_conn: AsyncConnT | None
    _async_lock: asyncio.Lock
    _closing_tasks: set[asyncio.Task[None]]
    _owns_connection: bool
    conn: SyncConnT

    def __init__(
        self,
        db_path: str,
        db_kwargs: dict[str, DBKwargValue],
        *,
        read_only: bool = False,
    ) -> None:
        """Initialize the database adapter and establish initial connection.

        Args:
            db_path: The file path or connection URI for the database.
            db_kwargs: Additional connection keyword arguments.
            read_only: Flag indicating whether mutating queries are disallowed.

        """
        self.db_path = db_path
        self.db_kwargs = db_kwargs
        self.read_only = read_only
        self._async_conn = None
        self._async_lock = asyncio.Lock()
        self._closing_tasks = set()
        self._owns_connection = not (("existing_conn" in db_kwargs and db_kwargs["existing_conn"] is not None) or ("conn" in db_kwargs and db_kwargs["conn"] is not None))
        self.conn = self.connect()

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

    async def connect_async(self) -> AsyncConnT:
        """Connect asynchronously to the underlying database.

        Returns:
            The active asynchronous database connection object.

        """
        res: AsyncConnT = await asyncio.to_thread(self.connect)  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]
        return res

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL statements to initialize the database schema.

        Args:
            ddl: The SQL DDL query string.

        """
        self.execute_with_feedback(ddl)

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
        except self.error_classes + (DatabaseExecutionError,) as e:
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
        except self.error_classes + (DatabaseExecutionError,) as e:
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
            try:
                cursor = self.conn.cursor()  # pyright: ignore[reportAttributeAccessIssue]
            except AttributeError:
                raw_rows = self.conn.execute(query, params or ()).fetchall()  # pyright: ignore[reportAttributeAccessIssue]
                return [tuple(r) for r in raw_rows]
            cursor.execute(query, params or ())
            desc = None
            with contextlib.suppress(AttributeError):
                desc = cursor.description
            results: list[tuple[JSONPrimitive, ...]] = [tuple(r) for r in cursor.fetchall()] if desc is not None else []
            with contextlib.suppress(AttributeError):
                cursor.close()
            return results
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

    def _schedule_or_run_async_close(self, cached_conn: AsyncConnT | None) -> None:
        """Safely close a cached asynchronous connection from a synchronous context.

        Args:
            cached_conn: Cached asynchronous connection object with a `.close()` method.

        """
        if cached_conn is None:
            return
        res = None
        with contextlib.suppress(AttributeError):
            res = cached_conn.close()  # pyright: ignore[reportAttributeAccessIssue]
        if asyncio.iscoroutine(res):
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(res)
                self._closing_tasks.add(task)
                task.add_done_callback(self._closing_tasks.discard)
            except RuntimeError:
                with contextlib.suppress(RuntimeError, OSError):
                    asyncio.run(res)

    def close(self) -> None:
        """Close the active synchronous connection and safely close or schedule any cached async connection."""
        cached = self._async_conn
        self._async_conn = None
        self._schedule_or_run_async_close(cached)
        if self._owns_connection:
            with contextlib.suppress(AttributeError):
                self.conn.close()  # pyright: ignore[reportAttributeAccessIssue]

    async def close_async(self) -> None:
        """Asynchronously close any cached async connection, pending close tasks, and underlying connection."""
        await self._await_closing_tasks()
        cached = self._async_conn
        self._async_conn = None
        if cached is not None:
            with contextlib.suppress(AttributeError):
                res = cached.close()  # pyright: ignore[reportAttributeAccessIssue]
                if asyncio.iscoroutine(res):
                    await res
        self.close()


__all__ = ["DBKwargValue", "DatabaseAdapter"]
