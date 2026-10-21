"""PostgreSQL database adapter supporting psycopg2 and asyncpg."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from gemma_4_sql.backends.lazy_loader import LazyLoader
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError

from .base import DatabaseAdapter

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONPrimitive

logger = logging.getLogger(__name__)
psycopg2 = LazyLoader("psycopg2").get_module()
asyncpg = LazyLoader("asyncpg").get_module()


class PostgresAdapter(DatabaseAdapter):
    """Adapter for PostgreSQL databases using `psycopg2` and `asyncpg`."""

    def __init__(
        self,
        db_path: str = "",
        db_kwargs: dict[str, object] | None = None,
        *,
        read_only: bool = True,
    ) -> None:
        """Initialize the PostgreSQL database adapter.

        Args:
            db_path: Connection DSN or URI string.
            db_kwargs: Optional connection keyword arguments.
            read_only: Whether to enforce read-only query execution.

        """
        self._async_conn: Any = None
        super().__init__(db_path, db_kwargs or {}, read_only=read_only)

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Provide the exception classes for PostgreSQL errors.

        Returns:
            Tuple of PostgreSQL driver exception types.

        """
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
        """Connect synchronously to PostgreSQL via `psycopg2`.

        Returns:
            Active `psycopg2` connection object.

        Raises:
            DependencyMissingError: If `psycopg2` is not installed.

        """
        if psycopg2 is None:
            msg = "psycopg2 is required. Install with `pip install psycopg2-binary`."
            raise DependencyMissingError(msg, dependency_name="psycopg2", backend="postgresql")
        if self.db_path and self.db_path != ":memory:":
            return psycopg2.connect(self.db_path, **self.db_kwargs)
        return psycopg2.connect(**self.db_kwargs)

    async def connect_async(self) -> object:
        """Connect asynchronously to PostgreSQL via `asyncpg`, reusing any cached connection.

        Returns:
            Active `asyncpg` connection object.

        Raises:
            DependencyMissingError: If `asyncpg` is not installed.

        """
        if asyncpg is None:
            msg = "asyncpg is required."
            raise DependencyMissingError(msg, dependency_name="asyncpg", backend="postgresql")
        cached = getattr(self, "_async_conn", None)
        if cached is not None:
            return cached
        if self.db_path and self.db_path != ":memory:":
            conn = await asyncpg.connect(self.db_path, **self.db_kwargs)
        else:
            conn = await asyncpg.connect(**self.db_kwargs)
        self._async_conn = conn
        return conn

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL statements to initialize the PostgreSQL schema.

        Args:
            ddl: DDL SQL statements string.

        """
        conn_obj = self.conn
        cursor = conn_obj.cursor()
        try:
            cursor.execute(ddl)
            conn_obj.commit()
        finally:
            cursor.close()

    async def execute_with_feedback_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute a PostgreSQL query asynchronously and return status, rows, and error feedback.

        Args:
            query: The SQL query string.
            params: Optional query parameters.

        Returns:
            A tuple of `(success, rows, error_message)`.

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
        """Execute a PostgreSQL query asynchronously via `asyncpg`.

        Args:
            query: The SQL query string.
            params: Optional query parameters.

        Returns:
            A list of result row tuples.

        Raises:
            DatabaseExecutionError: If PostgreSQL query execution fails.

        """
        try:
            async_conn = cast(Any, await self.connect_async())
            query_params = params or ()
            records = await async_conn.fetch(query, *query_params)
            return [tuple(r.values()) for r in records]
        except self.error_classes as e:
            logger.debug("Async Query execution failed: %s", e)
            raise DatabaseExecutionError(str(e), cause=e) from e

    async def close_async(self) -> None:
        """Close the cached asynchronous `asyncpg` connection if open."""
        cached = getattr(self, "_async_conn", None)
        if cached is not None:
            await cached.close()
        self._async_conn = None

    def close(self) -> None:
        """Close the synchronous database connection and close or schedule closure of cached async state."""
        cached = getattr(self, "_async_conn", None)
        self._async_conn = None
        if cached is not None and hasattr(cached, "close"):
            import asyncio
            from contextlib import suppress

            with suppress(RuntimeError):
                loop = asyncio.get_running_loop()
                loop.create_task(cached.close())
        super().close()
