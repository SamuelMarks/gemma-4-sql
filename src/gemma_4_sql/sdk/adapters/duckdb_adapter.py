"""DuckDB database adapter supporting in-process analytical SQL execution."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Sequence
from enum import Enum
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.lazy_loader import LazyLoader, ModuleNamespaceProtocol
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.type_hints import DatabaseConnectionProtocol, DuckDBConnectionProtocol, JSONPrimitive

from .base import (
    CursorFactoryConnectionProtocol,
    DatabaseAdapter,
    DBKwargValue,
    DirectExecConnectionProtocol,
    FetchableCursorProtocol,
    SyncAdapterConnection,
)

logger = logging.getLogger(__name__)


class _LoaderSentinel(Enum):
    """Sentinel indicating `duckdb` has not yet been lazily imported."""

    UNRESOLVED = "unresolved"


@runtime_checkable
class _DuckDBErrorModuleProtocol(Protocol):
    """Protocol for a `duckdb` module exposing `Error`."""

    @property
    def Error(self) -> type[BaseException] | int | str | None:
        """Provide `duckdb.Error` exception class."""
        ...


@runtime_checkable
class DuckDBModuleProtocol(_DuckDBErrorModuleProtocol, Protocol):
    """Protocol for the `duckdb` Python driver module."""

    def connect(
        self,
        database: str = ":memory:",
        *,
        read_only: bool = False,
        **kwargs: DBKwargValue,
    ) -> DuckDBConnectionProtocol | DirectExecConnectionProtocol:
        """Open a DuckDB database connection.

        Args:
            database: Database file path or `':memory:'`.
            read_only: Whether to open the database in read-only mode.
            **kwargs: Additional DuckDB connection parameters.

        Returns:
            Opened `DuckDBConnectionProtocol` connection.

        """
        ...


_DUCKDB_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("duckdb")
duckdb: DuckDBModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None = _LoaderSentinel.UNRESOLVED


def _resolve_duckdb(
    override: DuckDBModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None = _LoaderSentinel.UNRESOLVED,
) -> DuckDBModuleProtocol | ModuleNamespaceProtocol | None:
    """Resolve `duckdb` lazily while honoring explicit DI or module-level overrides.

    Args:
        override: Optional explicitly injected `duckdb` module or `None`.

    Returns:
        Loaded `duckdb` module or `None` if unavailable.

    """
    global duckdb
    if override is not _LoaderSentinel.UNRESOLVED:
        return override
    if duckdb is _LoaderSentinel.UNRESOLVED:
        duckdb = _DUCKDB_LOADER.get_module()
    return duckdb


class DuckDBAdapter(DatabaseAdapter[SyncAdapterConnection, SyncAdapterConnection]):
    """Adapter for DuckDB analytical databases."""

    _lock: threading.Lock
    _duckdb_override: DuckDBModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None

    def __init__(
        self,
        db_path: str = ":memory:",
        db_kwargs: dict[str, DBKwargValue] | None = None,
        *,
        read_only: bool = False,
        lazy_connect: bool = False,
        duckdb_module: (
            DuckDBModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None
        ) = _LoaderSentinel.UNRESOLVED,
    ) -> None:
        """Initialize the DuckDB adapter with a thread synchronization lock.

        Args:
            db_path: Path to DuckDB database file or `':memory:'`.
            db_kwargs: Optional connection keyword arguments.
            read_only: Whether to open the database in read-only mode.
            lazy_connect: Whether to defer synchronous connection establishment until `.conn` access.
            duckdb_module: Optional injected `duckdb` driver module or `None` for testing.

        """
        self._lock = threading.Lock()
        self._duckdb_override = duckdb_module
        super().__init__(db_path, db_kwargs or {}, read_only=read_only, lazy_connect=lazy_connect)

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Provide the exception classes for DuckDB errors.

        Returns:
            Tuple containing `duckdb.Error` or fallback execution error classes.

        """
        mod = _resolve_duckdb(self._duckdb_override)
        if isinstance(mod, _DuckDBErrorModuleProtocol):
            err_attr = mod.Error
            if isinstance(err_attr, type) and issubclass(err_attr, Exception):
                return (err_attr,)
        return (RuntimeError, OSError, DatabaseExecutionError)

    def connect(self) -> SyncAdapterConnection:
        """Connect synchronously to DuckDB or reuse an injected connection.

        Returns:
            Active DuckDB connection object.

        Raises:
            DependencyMissingError: If `duckdb` is not installed.

        """
        candidate = self.db_kwargs.get("existing_conn") or self.db_kwargs.get("conn")
        if isinstance(candidate, (DatabaseConnectionProtocol, DirectExecConnectionProtocol)):
            return candidate
        mod = _resolve_duckdb(self._duckdb_override)
        if not isinstance(mod, DuckDBModuleProtocol):
            msg = "duckdb is required. Install with `pip install duckdb`."
            raise DependencyMissingError(msg, dependency_name="duckdb", backend="duckdb")
        kwargs = self.db_kwargs.copy()
        ro_flag = bool(kwargs.pop("read_only", False)) or (self.read_only and self.db_path != ":memory:")
        raw_conn = mod.connect(self.db_path, read_only=ro_flag, **kwargs)
        if self.read_only:
            try:
                raw_conn.execute("SET enable_external_access = false;")
            except self.error_classes:
                pass
        return raw_conn

    async def connect_async(self) -> SyncAdapterConnection:
        """Return the active DuckDB connection for async operations.

        Returns:
            Active DuckDB connection object.

        """
        return self.conn

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL statements to initialize the DuckDB schema.

        Args:
            ddl: Schema DDL string to execute.

        Raises:
            DatabaseExecutionError: If DuckDB DDL execution fails.

        """
        if ddl and ddl.strip():
            with self._lock:
                try:
                    active = self.conn
                    if isinstance(active, DirectExecConnectionProtocol):
                        active.execute(ddl)
                    else:
                        active.cursor().execute(ddl)
                except self.error_classes as e:
                    raise DatabaseExecutionError(str(e), cause=e) from e

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
                    active_conn = self.conn
                    if isinstance(active_conn, CursorFactoryConnectionProtocol):
                        cur = active_conn.cursor()
                        return [tuple(r) for r in cur.execute(query, params or ()).fetchall()]
                    res: FetchableCursorProtocol = active_conn.execute(query, params or ())
                    return [tuple(r) for r in res.fetchall()]

            return await asyncio.to_thread(_exec)
        except self.error_classes as e:
            logger.debug("Async Query execution failed: %s", e)
            raise DatabaseExecutionError(str(e), cause=e) from e


__all__ = ["DuckDBAdapter", "DuckDBModuleProtocol"]
