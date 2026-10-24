"""DuckDB database adapter supporting in-process analytical SQL execution."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, cast

from gemma_4_sql.backends.lazy_loader import LazyLoader
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError

from .base import DatabaseAdapter

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONPrimitive

logger = logging.getLogger(__name__)
duckdb = LazyLoader("duckdb").get_module()


class DuckDBAdapter(DatabaseAdapter):
    """Adapter for DuckDB analytical databases."""

    def __init__(
        self,
        db_path: str = ":memory:",
        db_kwargs: dict[str, object] | None = None,
        *,
        read_only: bool = False,
    ) -> None:
        """Initialize the DuckDB adapter with a thread synchronization lock.

        Args:
            db_path: Path to DuckDB database file or `':memory:'`.
            db_kwargs: Optional connection keyword arguments.
            read_only: Whether to open the database in read-only mode.

        """
        self._lock = threading.Lock()
        super().__init__(db_path, db_kwargs or {}, read_only=read_only)

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Provide the exception classes for DuckDB errors.

        Returns:
            Tuple containing `duckdb.Error`.

        """
        import duckdb as _duckdb

        return (_duckdb.Error,)

    def connect(self) -> object:
        """Connect synchronously to DuckDB or reuse an injected connection.

        Returns:
            Active DuckDB connection object.

        Raises:
            DependencyMissingError: If `duckdb` is not installed.

        """
        if "existing_conn" in self.db_kwargs and self.db_kwargs["existing_conn"] is not None:
            return self.db_kwargs["existing_conn"]
        if "conn" in self.db_kwargs and self.db_kwargs["conn"] is not None:
            return self.db_kwargs["conn"]
        if duckdb is None:
            msg = "duckdb is required. Install with `pip install duckdb`."
            raise DependencyMissingError(msg, dependency_name="duckdb", backend="duckdb")
        kwargs = self.db_kwargs.copy()
        if self.read_only and self.db_path != ":memory:":
            kwargs["read_only"] = True
        return duckdb.connect(self.db_path, **kwargs)

    async def connect_async(self) -> object:
        """Return the active DuckDB connection for async operations.

        Returns:
            Active DuckDB connection object.

        """
        return self.conn

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL statements to initialize the DuckDB schema.

        Args:
            ddl: Schema DDL string to execute.

        """
        if "existing_conn" in self.db_kwargs or "conn" in self.db_kwargs:
            return
        self.conn.execute(ddl)

    def close(self) -> None:
        """Close the DuckDB connection if it was not externally provided."""
        if "existing_conn" in self.db_kwargs or "conn" in self.db_kwargs:
            return
        super().close()

    async def execute_with_feedback_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute a DuckDB query asynchronously and return status, rows, and error feedback.

        Args:
            query: SQL query string to execute.
            params: Optional query parameters.

        Returns:
            A tuple of `(success, rows, error_message)`.

        """
        try:
            results = await self.execute_query_async(query, params)
        except self.error_classes + (DatabaseExecutionError,) as e:
            cause = e.cause if isinstance(e, DatabaseExecutionError) and e.cause is not None else e
            return (False, [], str(cause))
        else:
            return (True, results, None)

    async def execute_query_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a DuckDB query asynchronously on a thread-safe cursor.

        Args:
            query: SQL query string to execute.
            params: Optional query parameters.

        Returns:
            List of result row tuples.

        Raises:
            DatabaseExecutionError: If DuckDB query execution fails.

        """
        try:
            loop = asyncio.get_running_loop()

            def _exec() -> list[tuple[JSONPrimitive, ...]]:
                """Execute query on a synchronized cursor.

                Returns:
                    Query result rows.

                """
                with self._lock:
                    conn_obj = self.conn
                    cur = conn_obj.cursor() if callable(getattr(conn_obj, "cursor", None)) else conn_obj
                    return cast("list[tuple[JSONPrimitive, ...]]", cur.execute(query, params or ()).fetchall())

            return await loop.run_in_executor(None, _exec)
        except self.error_classes as e:
            logger.debug("Async Query execution failed: %s", e)
            raise DatabaseExecutionError(str(e), cause=e) from e
