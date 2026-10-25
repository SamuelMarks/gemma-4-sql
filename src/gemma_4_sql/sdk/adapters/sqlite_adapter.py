"""SQLite database adapter with synchronous and asynchronous execution."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.lazy_loader import LazyLoader
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError

from .base import DatabaseAdapter

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONPrimitive

logger = logging.getLogger(__name__)
aiosqlite = LazyLoader("aiosqlite").get_module()


class SQLiteAdapter(DatabaseAdapter):
    """Adapter for SQLite databases supporting shared in-memory and file-backed URIs."""

    def __init__(
        self,
        db_path: str = ":memory:",
        db_kwargs: dict[str, object] | None = None,
        *,
        read_only: bool = True,
    ) -> None:
        """Initialize SQLite adapter with shared-memory support for in-memory databases.

        Args:
            db_path: Database file path or ':memory:'.
            db_kwargs: Optional database connection keyword arguments.
            read_only: Read-only enforcement flag.

        """
        self._actual_path = f"file:mem_{id(self)}?mode=memory&cache=shared" if db_path == ":memory:" else db_path
        self._async_conn: Any = None
        self._async_lock = asyncio.Lock()
        self._closing_tasks: set[Any] = set()
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
        kwargs: dict[str, Any] = dict(self.db_kwargs)
        if self.db_path == ":memory:":
            kwargs["uri"] = True
            kwargs.setdefault("check_same_thread", False)
            conn_mem: sqlite3.Connection = sqlite3.connect(self._actual_path, **kwargs)
            return conn_mem
        conn_file: sqlite3.Connection = sqlite3.connect(self.db_path, **kwargs)
        return conn_file

    async def connect_async(self) -> object:
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
                self._async_conn = await aiosqlite.connect(self._actual_path, **kwargs)
            else:
                self._async_conn = await aiosqlite.connect(self.db_path, **self.db_kwargs)
            return self._async_conn

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL statements to set up the SQLite schema.

        Args:
            ddl: DDL string containing SQL statements to execute.

        """
        conn_obj = self.conn
        with conn_obj:
            conn_obj.executescript(ddl)

    async def execute_with_feedback_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute a SQLite query asynchronously and return status, rows, and error feedback.

        Args:
            query: SQL query string.
            params: Optional tuple of query parameters.

        Returns:
            A tuple of `(success, fetched_rows, optional_error_message)`.

        """
        try:
            rows = await self.execute_query_async(query, params)
        except self.error_classes + (DatabaseExecutionError,) as e:
            cause = e.cause if isinstance(e, DatabaseExecutionError) and e.cause is not None else e
            return (False, [], str(cause))
        else:
            return (True, rows, None)

    async def execute_query_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a SQLite query asynchronously and return fetched rows.

        Args:
            query: SQL query string.
            params: Optional tuple of query parameters.

        Returns:
            List of result row tuples.

        Raises:
            DatabaseExecutionError: If SQLite execution fails.

        """
        try:
            async_conn: Any = await self.connect_async()
            try:
                cursor = await async_conn.execute(query, params or ())
                try:
                    if getattr(cursor, "description", None) is not None:
                        fetched = await cursor.fetchall()
                        return [tuple(r) for r in fetched]
                    if hasattr(async_conn, "commit"):
                        res_c = async_conn.commit()
                        if asyncio.iscoroutine(res_c):
                            await res_c
                    return []
                finally:
                    if hasattr(cursor, "close"):
                        res = cursor.close()
                        if asyncio.iscoroutine(res):
                            await res
            except (sqlite3.Error, RuntimeError, ValueError, TypeError, OSError):
                if hasattr(async_conn, "rollback"):
                    res_r = async_conn.rollback()
                    if asyncio.iscoroutine(res_r):
                        await res_r
                raise
        except self.error_classes as e:
            logger.debug("Async Query execution failed: %s", e)
            raise DatabaseExecutionError(str(e), cause=e) from e

    async def close_async(self) -> None:
        """Close the cached asynchronous `aiosqlite` connection and await any pending close tasks."""
        if self._closing_tasks:
            pending = list(self._closing_tasks)
            self._closing_tasks.clear()
            await asyncio.gather(*pending, return_exceptions=True)
        cached = self._async_conn
        self._async_conn = None
        if cached is not None and hasattr(cached, "close"):
            res = cached.close()
            if asyncio.iscoroutine(res):
                await res
        super().close()

    def close(self) -> None:
        """Close the synchronous database connection and safely close any cached async connection."""
        cached = self._async_conn
        self._async_conn = None
        if cached is not None and hasattr(cached, "close"):
            from contextlib import suppress

            res = cached.close()
            if asyncio.iscoroutine(res):
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(res)
                    self._closing_tasks.add(task)
                    task.add_done_callback(self._closing_tasks.discard)
                except RuntimeError:
                    with suppress(RuntimeError, OSError):
                        asyncio.run(res)
        super().close()
