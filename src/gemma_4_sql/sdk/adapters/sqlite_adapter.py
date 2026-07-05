# Copyright 2024
"""SQLite adapter."""

from __future__ import annotations

import logging
import sqlite3

from gemma_4_sql.backends.lazy_loader import LazyLoader

from .base import DatabaseAdapter

logger = logging.getLogger(__name__)
aiosqlite = LazyLoader("aiosqlite").get_module()


class SQLiteAdapter(DatabaseAdapter):
    """Adapter for SQLite."""

    def connect(self) -> sqlite3.Connection:
        """Connect synchronously.

        Returns:
            object: The resulting output from the operation.

        """
        return sqlite3.connect(self.db_path, **self.db_kwargs)

    async def connect_async(self) -> sqlite3.Connection:
        """Connect asynchronously.

        Returns:
            object: The resulting output from the operation.

        Raises:
        ImportError: If the operation encounters an unexpected ImportError.

        """
        if aiosqlite is None:
            msg = "aiosqlite is required."
            raise ImportError(msg)
        return await aiosqlite.connect(self.db_path, **self.db_kwargs)

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL to set up schema."""
        with self.conn:  # pragma: no cover
            self.conn.executescript(ddl)  # pragma: no cover

    async def execute_with_feedback_async(self, query: str, params: tuple[object, ...] | None = None) -> tuple[bool, list[tuple[object, ...]], str | None]:
        """Execute asynchronously with feedback.

        Returns:
            object: The resulting output from the operation.

        """
        try:
            async_conn = await self.connect_async()
            try:
                cursor = await async_conn.execute(query, params or ())
                try:
                    if getattr(cursor, "description", None) is not None:
                        results = await cursor.fetchall()
                        return (True, results, None)
                    return (True, [], None)
                finally:
                    await cursor.close()
            finally:
                if hasattr(async_conn, "close"):
                    await async_conn.close()
        except Exception as e:  # noqa: BLE001
            return (False, [], str(e))

    async def execute_query_async(self, query: str, params: tuple[object, ...] | None = None) -> list[tuple[object, ...]]:
        """Execute asynchronously.

        Returns:
            object: The resulting output from the operation.

        """
        try:
            async_conn = await self.connect_async()
            try:
                cursor = await async_conn.execute(query, params or ())
                try:
                    if getattr(cursor, "description", None) is not None:
                        return await cursor.fetchall()
                    return []
                finally:
                    await cursor.close()
            finally:
                if hasattr(async_conn, "close"):
                    await async_conn.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("Async Query execution failed: %s", e)
            return []
