"""PostgreSQL adapter."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONPrimitive

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
        classes: list[type[Exception]] = []
        if psycopg2 is not None:
            err = getattr(psycopg2, "Error", None)
            if isinstance(err, type) and issubclass(err, Exception):
                classes.append(err)
        if asyncpg is not None:
            err = getattr(asyncpg, "PostgresError", None)
            if isinstance(err, type) and issubclass(err, Exception):
                classes.append(err)
        if not classes:
            classes.append(Exception)
        return tuple(classes)

    def connect(self) -> object:
        """Connect synchronously.

        Returns:
            The execution result.

        Raises:
            ImportError: If psycopg2 is not installed.
        """
        if psycopg2 is None:
            msg = "psycopg2 is required. Install with `pip install psycopg2-binary`."
            raise ImportError(msg)
        if self.db_path and self.db_path != ":memory:":
            return psycopg2.connect(self.db_path, **self.db_kwargs)
        return psycopg2.connect(**self.db_kwargs)

    async def connect_async(self) -> object:
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
        conn_obj = cast(Any, self.conn)
        cursor = conn_obj.cursor()
        try:
            cursor.execute(ddl)
            conn_obj.commit()
        finally:
            cursor.close()

    async def execute_with_feedback_async(self, query: str, params: tuple[object, ...] | None = None) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute asynchronously with feedback.

        Args:
            query: The SQL query.
            params: Optional query parameters.

        Returns:
            A tuple of success boolean, result tuples, and optional error message.
        """
        try:
            async_conn = cast(Any, await self.connect_async())
            try:
                query_params = params or ()
                records = await async_conn.fetch(query, *query_params)
                results: list[tuple[JSONPrimitive, ...]] = [tuple(r.values()) for r in records]
                return (True, results, None)
            finally:
                if hasattr(async_conn, "close"):
                    await async_conn.close()
        except self.error_classes as e:
            return (False, [], str(e))

    async def execute_query_async(self, query: str, params: tuple[object, ...] | None = None) -> list[tuple[JSONPrimitive, ...]]:
        """Execute asynchronously.

        Args:
            query: The SQL query.
            params: Optional query parameters.

        Returns:
            A list of result tuples.
        """
        try:
            async_conn = cast(Any, await self.connect_async())
            try:
                query_params = params or ()
                records = await async_conn.fetch(query, *query_params)
                return [tuple(r.values()) for r in records]
            finally:
                if hasattr(async_conn, "close"):
                    await async_conn.close()
        except self.error_classes as e:
            logger.debug("Async Query execution failed: %s", e)
            return []
