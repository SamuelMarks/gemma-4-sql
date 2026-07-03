"""PostgreSQL adapter."""

from __future__ import annotations

import logging

from gemma_4_sql.backends.lazy_loader import LazyLoader

from .base import DatabaseAdapter

logger = logging.getLogger(__name__)
psycopg2 = LazyLoader("psycopg2").get_module()
asyncpg = LazyLoader("asyncpg").get_module()


class PostgresAdapter(DatabaseAdapter):
    """Adapter for PostgreSQL."""

    def connect(self) -> psycopg2.extensions.connection:
        """Connect synchronously."""
        if psycopg2 is None:
            msg = "psycopg2 is required. Install with `pip install psycopg2-binary`."
            raise ImportError(msg)
        if self.db_path and self.db_path != ":memory:":
            return psycopg2.connect(self.db_path, **self.db_kwargs)
        return psycopg2.connect(**self.db_kwargs)

    async def connect_async(self) -> psycopg2.extensions.connection:
        """Connect asynchronously."""
        if asyncpg is None:
            msg = "asyncpg is required."
            raise ImportError(msg)
        if self.db_path and self.db_path != ":memory:":
            return await asyncpg.connect(self.db_path, **self.db_kwargs)
        return await asyncpg.connect(**self.db_kwargs)

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL to set up schema."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(ddl)
            self.conn.commit()
        finally:
            cursor.close()

    def execute_with_feedback(self, query: str) -> tuple[bool, list[tuple[object, ...]], str | None]:
        """Execute synchronously with feedback."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
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
                records = await async_conn.fetch(query)
                results = [tuple(r.values()) for r in records]
                return (True, results, None)
            finally:
                if hasattr(async_conn, "close"):
                    await async_conn.close()
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            if e.__class__.__name__ in ("Error", "DatabaseError", "DataError", "ProgrammingError", "OperationalError", "IntegrityError", "InternalError", "NotSupportedError"):
                pass
            return (False, [], str(e))

    def execute_query(self, query: str) -> list[tuple[object, ...]]:
        """Execute synchronously."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
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
                records = await async_conn.fetch(query)
                return [tuple(r.values()) for r in records]
            finally:
                if hasattr(async_conn, "close"):
                    await async_conn.close()
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            if e.__class__.__name__ in ("Error", "DatabaseError", "DataError", "ProgrammingError", "OperationalError", "IntegrityError", "InternalError", "NotSupportedError"):
                pass
            logger.debug("Async Query execution failed: %s", e)
            return []
