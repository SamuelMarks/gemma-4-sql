"""SQLite adapter."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import TYPE_CHECKING, Any, cast

from gemma_4_sql.backends.lazy_loader import LazyLoader

from .base import DatabaseAdapter

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONPrimitive

logger = logging.getLogger(__name__)
aiosqlite = LazyLoader("aiosqlite").get_module()


class SQLiteAdapter(DatabaseAdapter):
    """Adapter for SQLite."""

    def __init__(
        self,
        db_path: str = ":memory:",
        db_kwargs: dict[str, object] | None = None,
        *,
        read_only: bool = True,
    ) -> None:
        """Initialize SQLite adapter with shared-memory support for in-memory databases.

        Args:
            db_path: Database path or ':memory:'.
            db_kwargs: Optional database connection keyword arguments.
            read_only: Read-only enforcement flag.
        """
        self._actual_path = f"file:mem_{id(self)}?mode=memory&cache=shared" if db_path == ":memory:" else db_path
        super().__init__(db_path, db_kwargs or {}, read_only=read_only)

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Return the exception classes for SQLite errors.

        Returns:
            Tuple of handled SQLite exception types.
        """
        return (sqlite3.Error,)

    def connect(self) -> sqlite3.Connection:
        """Connect synchronously to SQLite.

        Returns:
            A sqlite3.Connection instance.
        """
        kwargs = dict(self.db_kwargs)
        if self.db_path == ":memory:":
            kwargs["uri"] = True
            kwargs.setdefault("check_same_thread", False)
            return sqlite3.connect(self._actual_path, **cast(dict[str, Any], kwargs))
        return sqlite3.connect(self.db_path, **cast(dict[str, Any], self.db_kwargs))

    async def connect_async(self) -> object:
        """Connect asynchronously to SQLite.

        Returns:
            The aiosqlite connection instance.

        Raises:
            ImportError: If aiosqlite is missing for async database connections.
        """
        if aiosqlite is None:
            msg = "aiosqlite is required."
            raise ImportError(msg)
        kwargs = dict(self.db_kwargs)
        if self.db_path == ":memory:":
            kwargs["uri"] = True
            return await aiosqlite.connect(self._actual_path, **kwargs)
        return await aiosqlite.connect(self.db_path, **self.db_kwargs)

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL to set up schema.

        Args:
            ddl: DDL string containing SQL statements to execute.
        """
        conn_obj = cast(Any, self.conn)
        with conn_obj:
            conn_obj.executescript(ddl)

    async def execute_with_feedback_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute asynchronously with feedback.

        Args:
            query: SQL query string.
            params: Optional tuple of query parameters.

        Returns:
            A tuple of (success, fetched rows, optional error message).
        """
        try:
            async_conn = cast(Any, await self.connect_async())
            try:
                cursor = await async_conn.execute(query, params or ())
                try:
                    if getattr(cursor, "description", None) is not None:
                        results = await cursor.fetchall()
                        return (True, results, None)
                    return (True, [], None)
                finally:
                    if hasattr(cursor, "close"):
                        res = cursor.close()
                        if asyncio.iscoroutine(res):
                            await res
            finally:
                if hasattr(async_conn, "close"):
                    res = async_conn.close()
                    if asyncio.iscoroutine(res):
                        await res
        except self.error_classes as e:
            return (False, [], str(e))

    async def execute_query_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute asynchronously.

        Args:
            query: SQL query string.
            params: Optional tuple of query parameters.

        Returns:
            List of result rows.
        """
        try:
            async_conn = cast(Any, await self.connect_async())
            try:
                cursor = await async_conn.execute(query, params or ())
                try:
                    if getattr(cursor, "description", None) is not None:
                        return await cursor.fetchall()
                    return []
                finally:
                    if hasattr(cursor, "close"):
                        res = cursor.close()
                        if asyncio.iscoroutine(res):
                            await res
            finally:
                if hasattr(async_conn, "close"):
                    res = async_conn.close()
                    if asyncio.iscoroutine(res):
                        await res
        except self.error_classes as e:
            logger.debug("Async Query execution failed: %s", e)
            return []
