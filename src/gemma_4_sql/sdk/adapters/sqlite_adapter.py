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
        """Connect synchronously."""
        return sqlite3.connect(self.db_path, **self.db_kwargs)

    async def connect_async(self) -> sqlite3.Connection:
        """Connect asynchronously."""
        if aiosqlite is None:
            msg = "aiosqlite is required."
            raise ImportError(msg)
        return await aiosqlite.connect(self.db_path, **self.db_kwargs)

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL to set up schema."""
        with self.conn:
            self.conn.executescript(ddl)

    def execute_with_feedback(self, query: str) -> tuple[bool, list[tuple[object, ...]], str | None]:
        """Execute synchronously with feedback."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
        except (sqlite3.Error, RuntimeError, ValueError, TypeError, AttributeError) as e:
            if e.__class__.__name__ in ("Error", "DatabaseError", "DataError", "ProgrammingError", "OperationalError", "IntegrityError", "InternalError", "NotSupportedError"):
                pass
            return (False, [], str(e))
        else:
            if cursor.description is not None:
                return (True, cursor.fetchall(), None)
            return (True, [], None)
        finally:
            if "cursor" in locals():
                cursor.close()

    async def execute_with_feedback_async(self, query: str) -> tuple[bool, list[tuple[object, ...]], str | None]:
        """Execute asynchronously with feedback."""
        try:
            async_conn = await self.connect_async()
            try:
                cursor = await async_conn.execute(query)
                try:
                    if cursor.description is not None:
                        results = await cursor.fetchall()
                        return (True, results, None)
                    return (True, [], None)
                finally:
                    await cursor.close()
            finally:
                if hasattr(async_conn, "close"):
                    await async_conn.close()
        except (sqlite3.Error, RuntimeError, ValueError, TypeError, AttributeError) as e:
            if e.__class__.__name__ in ("Error", "DatabaseError", "DataError", "ProgrammingError", "OperationalError", "IntegrityError", "InternalError", "NotSupportedError"):
                pass
            return (False, [], str(e))

    def execute_query(self, query: str) -> list[tuple[object, ...]]:
        """Execute synchronously."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
        except (sqlite3.Error, RuntimeError, ValueError, TypeError, AttributeError) as e:
            if e.__class__.__name__ in ("Error", "DatabaseError", "DataError", "ProgrammingError", "OperationalError", "IntegrityError", "InternalError", "NotSupportedError"):
                pass
            logger.debug("Query execution failed: %s", e)
            return []
        else:
            if cursor.description is not None:
                return cursor.fetchall()
            return []
        finally:
            if "cursor" in locals():
                cursor.close()

    async def execute_query_async(self, query: str) -> list[tuple[object, ...]]:
        """Execute asynchronously."""
        try:
            async_conn = await self.connect_async()
            try:
                cursor = await async_conn.execute(query)
                try:
                    if cursor.description is not None:
                        return await cursor.fetchall()
                    return []
                finally:
                    await cursor.close()
            finally:
                if hasattr(async_conn, "close"):
                    await async_conn.close()
        except (sqlite3.Error, RuntimeError, ValueError, TypeError, AttributeError) as e:
            if e.__class__.__name__ in ("Error", "DatabaseError", "DataError", "ProgrammingError", "OperationalError", "IntegrityError", "InternalError", "NotSupportedError"):
                pass
            logger.debug("Async Query execution failed: %s", e)
            return []
