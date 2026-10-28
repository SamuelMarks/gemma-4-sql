"""SQLite database adapter with synchronous and asynchronous execution."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import sqlite3
from collections.abc import Callable, Coroutine, Sequence
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.lazy_loader import LazyLoader
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.type_hints import AsyncDatabaseConnectionProtocol, JSONPrimitive

from .base import DatabaseAdapter, DBKwargValue

logger = logging.getLogger(__name__)


@runtime_checkable
class _AiosqliteCursorProtocol(Protocol):
    """Protocol for an `aiosqlite.Cursor` supporting `.fetchall()` and `.close()`."""

    @property
    def description(self) -> Sequence[Sequence[str | int | None]] | None:
        """Provide the cursor column description tuple."""
        ...

    async def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
        """Fetch all remaining rows."""
        ...

    def close(self) -> Coroutine[None, None, None] | None:
        """Close the cursor (synchronously or as a coroutine)."""
        ...


@runtime_checkable
class AiosqliteConnectionProtocol(AsyncDatabaseConnectionProtocol, Protocol):
    """Protocol for an `aiosqlite.Connection` supporting `.execute(...)`, `.commit()`, and `.rollback()`."""

    async def execute(
        self,
        sql: str,
        parameters: Sequence[JSONPrimitive] = (),
        /,
    ) -> _AiosqliteCursorProtocol:
        """Execute a SQL statement and return a cursor."""
        ...

    def commit(self) -> Coroutine[None, None, None] | None:
        """Commit the current transaction."""
        ...

    def rollback(self) -> Coroutine[None, None, None] | None:
        """Roll back the current transaction."""
        ...


@runtime_checkable
class AiosqliteModuleProtocol(Protocol):
    """Protocol for the `aiosqlite` Python driver module."""

    async def connect(self, database: str, **kwargs: DBKwargValue) -> AiosqliteConnectionProtocol:
        """Open an asynchronous `aiosqlite` connection."""
        ...


aiosqlite = LazyLoader("aiosqlite").get_module()
_SQLITE_MEM_COUNTER = itertools.count(1)


class SQLiteAdapter(DatabaseAdapter[sqlite3.Connection, AiosqliteConnectionProtocol]):
    """Adapter for SQLite databases supporting shared in-memory and file-backed URIs."""

    _actual_path: str

    def __init__(
        self,
        db_path: str = ":memory:",
        db_kwargs: dict[str, DBKwargValue] | None = None,
        *,
        read_only: bool = True,
    ) -> None:
        """Initialize SQLite adapter with shared-memory support for in-memory databases.

        Args:
            db_path: Database file path or `':memory:'`.
            db_kwargs: Optional database connection keyword arguments.
            read_only: Read-only enforcement flag.

        """
        self._actual_path = f"file:mem_{id(self)}_{next(_SQLITE_MEM_COUNTER)}?mode=memory&cache=shared" if db_path == ":memory:" else db_path
        clean_kwargs = dict(db_kwargs) if db_kwargs else {}
        clean_kwargs.pop("defer_ddl", None)
        super().__init__(db_path, clean_kwargs, read_only=read_only)

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Provide the exception classes for SQLite errors.

        Returns:
            Tuple of handled SQLite exception types.

        """
        return (sqlite3.Error,)

    def connect(self) -> sqlite3.Connection:
        """Connect synchronously to SQLite.

        Returns:
            An active `sqlite3.Connection` instance.

        """
        sqlite_connect: Callable[..., sqlite3.Connection] = sqlite3.connect
        kwargs: dict[str, DBKwargValue] = dict(self.db_kwargs)
        if self.db_path == ":memory:":
            kwargs["uri"] = True
            kwargs.setdefault("check_same_thread", False)
            return sqlite_connect(self._actual_path, **kwargs)  # ty: ignore[no-matching-overload]
        return sqlite_connect(self.db_path, **kwargs)  # ty: ignore[no-matching-overload]

    async def connect_async(self) -> AiosqliteConnectionProtocol:
        """Connect asynchronously to SQLite via `aiosqlite`, reusing any cached connection.

        Returns:
            The active `aiosqlite` connection instance.

        Raises:
            DependencyMissingError: If `aiosqlite` is not installed.

        """
        if aiosqlite is None:
            msg = "aiosqlite is required."
            raise DependencyMissingError(msg, dependency_name="aiosqlite", backend="sqlite")
        async with self._async_lock:
            if self._async_conn is not None:
                return self._async_conn
            kwargs = dict(self.db_kwargs)
            if self.db_path == ":memory:":
                kwargs["uri"] = True
                conn: AiosqliteConnectionProtocol = await aiosqlite.connect(self._actual_path, **kwargs)
            else:
                conn = await aiosqlite.connect(self.db_path, **self.db_kwargs)
            self._async_conn = conn
            return conn

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL statements to set up the SQLite schema.

        Args:
            ddl: DDL string containing SQL statements to execute.

        """
        conn_obj = self.conn
        with conn_obj:
            conn_obj.executescript(ddl)

    async def execute_query_async(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a SQLite query asynchronously and return fetched rows.

        Args:
            query: SQL query string.
            params: Optional sequence of query parameters.

        Returns:
            List of result row tuples.

        Raises:
            DatabaseExecutionError: If SQLite execution fails.

        """
        try:
            async_conn = await self.connect_async()
            try:
                cursor = await async_conn.execute(query, params or ())
                try:
                    desc = None
                    with contextlib.suppress(AttributeError):
                        desc = cursor.description
                    if desc is not None:
                        fetched = await cursor.fetchall()
                        return [tuple(r) for r in fetched]
                    with contextlib.suppress(AttributeError):
                        res_c = async_conn.commit()
                        if asyncio.iscoroutine(res_c):
                            await res_c
                    return []
                finally:
                    with contextlib.suppress(AttributeError):
                        res = cursor.close()
                        if asyncio.iscoroutine(res):
                            await res
            except (sqlite3.Error, RuntimeError, ValueError, TypeError, OSError):
                with contextlib.suppress(AttributeError):
                    res_r = async_conn.rollback()
                    if asyncio.iscoroutine(res_r):
                        await res_r
                raise
        except self.error_classes as e:
            logger.debug("Async Query execution failed: %s", e)
            raise DatabaseExecutionError(str(e), cause=e) from e


__all__ = ["AiosqliteConnectionProtocol", "AiosqliteModuleProtocol", "SQLiteAdapter"]
