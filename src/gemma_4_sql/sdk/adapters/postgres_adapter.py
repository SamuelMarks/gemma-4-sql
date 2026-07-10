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

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Return the exception classes."""
        classes = []
        if psycopg2 is not None:
            classes.append(psycopg2.Error)
        if asyncpg is not None:
            classes.append(asyncpg.PostgresError)
        if not classes:
            classes.append(Exception)
        return tuple(classes)

    def connect(self) -> psycopg2.extensions.connection:
        """Connect synchronously.

        Returns:
            The execution result.
        """
        if psycopg2 is None:
            msg = "psycopg2 is required. Install with `pip install psycopg2-binary`."
            raise ImportError(msg)
        if self.db_path and self.db_path != ":memory:":
            return psycopg2.connect(self.db_path, **self.db_kwargs)
        return psycopg2.connect(**self.db_kwargs)

    async def connect_async(self) -> psycopg2.extensions.connection:
        """Connect asynchronously.

        Returns:
            object: The resulting output from the operation.

        Raises:
        ImportError: If the operation encounters an unexpected ImportError.

        """
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

    async def execute_with_feedback_async(self, query: str, params: tuple[object, ...] | None = None) -> tuple[bool, list[tuple[object, ...]], str | None]:
        """Execute asynchronously with feedback.

        Returns:
            object: The resulting output from the operation.

        """
        try:
            async_conn = await self.connect_async()
            try:
                records = await async_conn.fetch(query)
                results = [tuple(r.values()) for r in records]
                return (True, results, None)
            finally:
                if hasattr(async_conn, "close"):
                    await async_conn.close()
        except self.error_classes as e:
            return (False, [], str(e))

    async def execute_query_async(self, query: str, params: tuple[object, ...] | None = None) -> list[tuple[object, ...]]:
        """Execute asynchronously.

        Returns:
            object: The resulting output from the operation.

        """
        try:
            async_conn = await self.connect_async()
            try:
                records = await async_conn.fetch(query)
                return [tuple(r.values()) for r in records]
            finally:
                if hasattr(async_conn, "close"):
                    await async_conn.close()
        except self.error_classes as e:
            logger.debug("Async Query execution failed: %s", e)
            return []
