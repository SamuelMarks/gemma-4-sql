"""PostgreSQL database adapter supporting psycopg2 and asyncpg."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.lazy_loader import LazyLoader
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.type_hints import (
    AsyncDatabaseConnectionProtocol,
    DatabaseConnectionProtocol,
    JSONPrimitive,
)

from .base import DatabaseAdapter, DBKwargValue

logger = logging.getLogger(__name__)


@runtime_checkable
class _AsyncpgRecordProtocol(Protocol):
    """Protocol for an `asyncpg.Record` exposing `.values()`."""

    def values(self) -> Sequence[JSONPrimitive]:
        """Return record column values."""
        ...


@runtime_checkable
class _AsyncpgConnectionProtocol(AsyncDatabaseConnectionProtocol, Protocol):
    """Protocol for an `asyncpg.Connection` supporting `.fetch(...)` and `.close()`."""

    async def fetch(self, query: str, *args: JSONPrimitive) -> Sequence[_AsyncpgRecordProtocol]:
        """Execute a SQL query and return fetched records."""
        ...


@runtime_checkable
class Psycopg2ModuleProtocol(Protocol):
    """Protocol for the `psycopg2` Python driver module."""

    @property
    def Error(self) -> type[Exception]:
        """Provide `psycopg2.Error` exception class."""
        ...

    def connect(self, dsn: str = "", **kwargs: DBKwargValue) -> DatabaseConnectionProtocol:
        """Open a synchronous `psycopg2` connection."""
        ...


@runtime_checkable
class AsyncpgModuleProtocol(Protocol):
    """Protocol for the `asyncpg` Python driver module."""

    @property
    def PostgresError(self) -> type[Exception]:
        """Provide `asyncpg.PostgresError` exception class."""
        ...

    async def connect(self, dsn: str = "", **kwargs: DBKwargValue) -> _AsyncpgConnectionProtocol:
        """Open an asynchronous `asyncpg` connection."""
        ...


psycopg2 = LazyLoader("psycopg2").get_module()
asyncpg = LazyLoader("asyncpg").get_module()


class PostgresAdapter(DatabaseAdapter[DatabaseConnectionProtocol, _AsyncpgConnectionProtocol]):
    """Adapter for PostgreSQL databases using `psycopg2` and `asyncpg`."""

    def __init__(
        self,
        db_path: str = "",
        db_kwargs: dict[str, DBKwargValue] | None = None,
        *,
        read_only: bool = True,
    ) -> None:
        """Initialize the PostgreSQL database adapter.

        Args:
            db_path: Connection DSN or URI string.
            db_kwargs: Optional connection keyword arguments.
            read_only: Whether to enforce read-only query execution.

        """
        super().__init__(db_path, db_kwargs or {}, read_only=read_only)

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Provide the exception classes for PostgreSQL errors.

        Returns:
            Tuple of PostgreSQL driver exception types.

        """
        classes: list[type[Exception]] = []
        if psycopg2 is not None:
            with contextlib.suppress(AttributeError):
                err = psycopg2.Error
                if isinstance(err, type) and issubclass(err, Exception):
                    classes.append(err)
        if asyncpg is not None:
            with contextlib.suppress(AttributeError):
                err = asyncpg.PostgresError
                if isinstance(err, type) and issubclass(err, Exception):
                    classes.append(err)
        if not classes:
            return (RuntimeError, OSError, DatabaseExecutionError)
        return tuple(classes)

    def connect(self) -> DatabaseConnectionProtocol:
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
            conn: DatabaseConnectionProtocol = psycopg2.connect(self.db_path, **self.db_kwargs)
            return conn
        conn_default: DatabaseConnectionProtocol = psycopg2.connect(**self.db_kwargs)
        return conn_default

    async def connect_async(self) -> _AsyncpgConnectionProtocol:
        """Connect asynchronously to PostgreSQL via `asyncpg`, reusing any cached connection.

        Returns:
            Active `asyncpg` connection object.

        Raises:
            DependencyMissingError: If `asyncpg` is not installed.

        """
        if asyncpg is None:
            msg = "asyncpg is required."
            raise DependencyMissingError(msg, dependency_name="asyncpg", backend="postgresql")
        async with self._async_lock:
            if self._async_conn is not None:
                return self._async_conn
            if self.db_path and self.db_path != ":memory:":
                conn: _AsyncpgConnectionProtocol = await asyncpg.connect(self.db_path, **self.db_kwargs)
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
            with contextlib.suppress(AttributeError):
                conn_obj.commit()
        except self.error_classes + (RuntimeError, ValueError, TypeError, OSError):
            with contextlib.suppress(AttributeError):
                conn_obj.rollback()
            raise
        finally:
            cursor.close()

    async def execute_query_async(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
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
            async_conn = await self.connect_async()
            query_params = params or ()
            records = await async_conn.fetch(query, *query_params)
            return [tuple(r.values()) for r in records]
        except self.error_classes as e:
            self._async_conn = None
            logger.debug("Async Query execution failed: %s", e)
            raise DatabaseExecutionError(str(e), cause=e) from e


__all__ = ["AsyncpgModuleProtocol", "PostgresAdapter", "Psycopg2ModuleProtocol"]
