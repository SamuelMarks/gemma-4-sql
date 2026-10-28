"""DuckDB database adapter supporting in-process analytical SQL execution."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.lazy_loader import LazyLoader
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.type_hints import DatabaseConnectionProtocol, DuckDBConnectionProtocol, JSONPrimitive

from .base import DatabaseAdapter, DBKwargValue

logger = logging.getLogger(__name__)


@runtime_checkable
class DuckDBModuleProtocol(Protocol):
    """Protocol for the `duckdb` Python driver module."""

    @property
    def Error(self) -> type[Exception]:
        """Provide `duckdb.Error` exception class."""
        ...

    def connect(
        self,
        database: str = ":memory:",
        *,
        read_only: bool = False,
        **kwargs: DBKwargValue,
    ) -> DuckDBConnectionProtocol:
        """Open a DuckDB database connection."""
        ...


duckdb = LazyLoader("duckdb").get_module()


class DuckDBAdapter(DatabaseAdapter[DatabaseConnectionProtocol, DatabaseConnectionProtocol]):
    """Adapter for DuckDB analytical databases."""

    _lock: threading.Lock

    def __init__(
        self,
        db_path: str = ":memory:",
        db_kwargs: dict[str, DBKwargValue] | None = None,
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
            Tuple containing `duckdb.Error` or fallback execution error classes.

        """
        with contextlib.suppress(AttributeError, TypeError):
            err: type[Exception] = duckdb.Error  # pyright: ignore[reportOptionalMemberAccess]  # ty: ignore[unresolved-attribute]
            return (err,)
        return (RuntimeError, OSError, DatabaseExecutionError)

    def connect(self) -> DatabaseConnectionProtocol:
        """Connect synchronously to DuckDB or reuse an injected connection.

        Returns:
            Active DuckDB connection object.

        Raises:
            DependencyMissingError: If `duckdb` is not installed.

        """
        if "existing_conn" in self.db_kwargs and self.db_kwargs["existing_conn"] is not None:
            return self.db_kwargs["existing_conn"]  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-return-type]
        if "conn" in self.db_kwargs and self.db_kwargs["conn"] is not None:
            return self.db_kwargs["conn"]  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-return-type]
        if duckdb is None:
            msg = "duckdb is required. Install with `pip install duckdb`."
            raise DependencyMissingError(msg, dependency_name="duckdb", backend="duckdb")
        kwargs = self.db_kwargs.copy()
        if self.read_only and self.db_path != ":memory:":
            kwargs["read_only"] = True
        conn: DatabaseConnectionProtocol = duckdb.connect(self.db_path, **kwargs)
        return conn

    async def connect_async(self) -> DatabaseConnectionProtocol:
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
        if ddl and ddl.strip():
            with self._lock:
                self.conn.execute(ddl)

    def close(self) -> None:
        """Close the DuckDB connection if it was not externally provided."""
        with self._lock:
            super().close()

    def execute_query(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a DuckDB query synchronously with thread-lock synchronization.

        Args:
            query: SQL query string to execute.
            params: Optional query parameters.

        Returns:
            List of result row tuples.

        Raises:
            DatabaseExecutionError: If DuckDB query execution fails.

        """
        with self._lock:
            return super().execute_query(query, params)

    async def execute_query_async(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
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

            def _exec() -> list[tuple[JSONPrimitive, ...]]:
                """Execute query on a synchronized cursor.

                Returns:
                    Query result rows.

                """
                with self._lock:
                    cur = self.conn
                    with contextlib.suppress(AttributeError, TypeError):
                        cur = self.conn.cursor()  # pyright: ignore[reportAssignmentType]
                    rows: list[tuple[JSONPrimitive, ...]] = [tuple(r) for r in cur.execute(query, params or ()).fetchall()]
                    return rows

            return await asyncio.to_thread(_exec)
        except self.error_classes as e:
            logger.debug("Async Query execution failed: %s", e)
            raise DatabaseExecutionError(str(e), cause=e) from e


__all__ = ["DuckDBAdapter", "DuckDBModuleProtocol"]
