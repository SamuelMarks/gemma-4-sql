"""SQLite database adapter with synchronous, asynchronous, and driver-level read-only execution."""

from __future__ import annotations

import asyncio
import itertools
import logging
import sqlite3
import threading
from collections.abc import Coroutine, Sequence
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import quote

from gemma_4_sql.backends.lazy_loader import LazyLoader, ModuleNamespaceProtocol
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.type_hints import AsyncDatabaseConnectionProtocol, JSONPrimitive

from .base import DatabaseAdapter, DBKwargValue

logger = logging.getLogger(__name__)


class _LoaderSentinel(Enum):
    """Sentinel indicating an optional driver module has not yet been lazily loaded."""

    UNRESOLVED = "unresolved"


@runtime_checkable
class _DescribableAsyncCursorProtocol(Protocol):
    """Protocol for an async SQLite cursor exposing `.description`."""

    @property
    def description(self) -> Sequence[Sequence[str | int | None]] | bool | None:
        """Provide the cursor column description tuple."""
        ...


@runtime_checkable
class _LoadExtToggleProtocol(Protocol):
    """Protocol for SQLite connections exposing `enable_load_extension`."""

    def enable_load_extension(self, enabled: bool, /) -> None:
        """Enable or disable loading C shared-library extensions.

        Args:
            enabled: Whether loadable extensions are permitted.

        """
        ...


@runtime_checkable
class _SyncExecutableProtocol(Protocol):
    """Protocol for synchronous SQLite connections exposing `.execute(...)`."""

    def execute(self, sql: str, /) -> sqlite3.Cursor | None:
        """Execute a synchronous SQL statement.

        Args:
            sql: SQL statement text.

        Returns:
            Optional `sqlite3.Cursor` handle.

        """
        ...


@runtime_checkable
class _ClosableAsyncCursorProtocol(Protocol):
    """Protocol for an async SQLite cursor exposing `.close()`."""

    def close(self) -> Coroutine[None, None, None] | None:
        """Close the cursor."""
        ...


@runtime_checkable
class _AiosqliteCursorProtocol(Protocol):
    """Protocol for an `aiosqlite.Cursor` supporting `.fetchall()`."""

    async def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
        """Fetch all remaining rows.

        Returns:
            Sequence of result rows.

        """
        ...


@runtime_checkable
class _CommittableAsyncConnProtocol(Protocol):
    """Protocol for an async connection exposing `.commit()`."""

    def commit(self) -> Coroutine[None, None, None] | None:
        """Commit the current transaction."""
        ...


@runtime_checkable
class _RollbackableAsyncConnProtocol(Protocol):
    """Protocol for an async connection exposing `.rollback()`."""

    def rollback(self) -> Coroutine[None, None, None] | None:
        """Roll back the current transaction."""
        ...


@runtime_checkable
class AiosqliteConnectionProtocol(AsyncDatabaseConnectionProtocol, Protocol):
    """Protocol for an `aiosqlite.Connection` supporting `.execute(...)`."""

    async def execute(
        self,
        sql: str,
        parameters: Sequence[JSONPrimitive] = (),
        /,
    ) -> _AiosqliteCursorProtocol:
        """Execute a SQL statement and return a cursor.

        Args:
            sql: SQL statement text.
            parameters: Positional bind parameters.

        Returns:
            Async cursor instance.

        """
        ...


@runtime_checkable
class AiosqliteModuleProtocol(Protocol):
    """Protocol for the `aiosqlite` Python driver module."""

    async def connect(self, database: str, **kwargs: DBKwargValue) -> AiosqliteConnectionProtocol:
        """Open an asynchronous `aiosqlite` connection.

        Args:
            database: Database path or SQLite URI string.
            **kwargs: Driver connection keyword arguments.

        Returns:
            Opened `AiosqliteConnectionProtocol` connection.

        """
        ...


_AIOSQLITE_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("aiosqlite")
aiosqlite: AiosqliteModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None = _LoaderSentinel.UNRESOLVED
_SQLITE_MEM_COUNTER = itertools.count(1)


def _resolve_aiosqlite(
    override: AiosqliteModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None = _LoaderSentinel.UNRESOLVED,
) -> AiosqliteModuleProtocol | None:
    """Resolve the `aiosqlite` module lazily while honoring explicit DI or module-attribute overrides.

    Args:
        override: Optional explicitly injected `aiosqlite` module or `None`.

    Returns:
        Typed `AiosqliteModuleProtocol` when available, otherwise `None`.

    """
    global aiosqlite
    target = aiosqlite if override is _LoaderSentinel.UNRESOLVED else override
    if target is _LoaderSentinel.UNRESOLVED:
        aiosqlite = _AIOSQLITE_LOADER.get_module()
        target = aiosqlite
    if isinstance(target, AiosqliteModuleProtocol):
        return target
    return None


def _build_sqlite_readonly_uri(db_path: str) -> tuple[str, bool]:
    """Construct a driver-level read-only SQLite URI when `db_path` refers to an existing file.

    Args:
        db_path: File path or URI string.

    Returns:
        Tuple of `(resolved_target_path, use_uri_flag)`.

    """
    if db_path.startswith("file:"):
        sep = "&" if "?" in db_path else "?"
        return (db_path if "mode=" in db_path else f"{db_path}{sep}mode=ro", True)
    path_obj = Path(db_path)
    if path_obj.exists():
        encoded = quote(str(path_obj.resolve()), safe="/:")
        return (f"file:{encoded}?mode=ro", True)
    return (db_path, False)


class SQLiteAdapter(DatabaseAdapter[sqlite3.Connection, AiosqliteConnectionProtocol]):
    """Adapter for SQLite databases supporting shared in-memory and driver-level read-only URIs."""

    _actual_path: str
    _lock: threading.Lock
    _aiosqlite_override: AiosqliteModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None

    def __init__(
        self,
        db_path: str = ":memory:",
        db_kwargs: dict[str, DBKwargValue] | None = None,
        *,
        read_only: bool = True,
        lazy_connect: bool = False,
        aiosqlite_module: (
            AiosqliteModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None
        ) = _LoaderSentinel.UNRESOLVED,
    ) -> None:
        """Initialize SQLite adapter with shared-memory support for in-memory databases.

        Args:
            db_path: Database file path or `':memory:'`.
            db_kwargs: Optional database connection keyword arguments.
            read_only: Read-only enforcement flag.
            lazy_connect: Whether to defer synchronous connection establishment until `.conn` access.
            aiosqlite_module: Optional injected `aiosqlite` driver module or `None` for testing.

        """
        self._lock = threading.Lock()
        self._aiosqlite_override = aiosqlite_module
        self._actual_path = (
            f"file:mem_{next(_SQLITE_MEM_COUNTER)}?mode=memory&cache=shared" if db_path == ":memory:" else db_path
        )
        clean_kwargs = dict(db_kwargs) if db_kwargs else {}
        clean_kwargs.pop("defer_ddl", None)
        super().__init__(db_path, clean_kwargs, read_only=read_only, lazy_connect=lazy_connect)

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Provide the exception classes for SQLite errors.

        Returns:
            Tuple of handled SQLite exception types.

        """
        return (sqlite3.Error,)

    def _harden_sqlite_connection(
        self,
        conn: sqlite3.Connection | _LoadExtToggleProtocol | _SyncExecutableProtocol,
        *,
        enforce_query_only: bool,
    ) -> None:
        """Disable SQLite loadable extensions and optionally enforce `PRAGMA query_only = ON`.

        Args:
            conn: Open SQLite connection instance.
            enforce_query_only: Whether to set `PRAGMA query_only = ON` on this connection.

        """
        if isinstance(conn, _LoadExtToggleProtocol):
            try:
                conn.enable_load_extension(False)
            except sqlite3.Error:
                pass
        if enforce_query_only and isinstance(conn, _SyncExecutableProtocol):
            conn.execute("PRAGMA query_only = ON;")

    def connect(self) -> sqlite3.Connection:
        """Connect synchronously to SQLite with extension loading disabled and read-only enforcement.

        Returns:
            An active `sqlite3.Connection` instance.

        """
        raw_timeout = self.db_kwargs.get("timeout", 5.0)
        timeout = float(raw_timeout) if isinstance(raw_timeout, (int, float)) else 5.0
        raw_detect = self.db_kwargs.get("detect_types", 0)
        detect_types = int(raw_detect) if isinstance(raw_detect, int) else 0
        raw_cached = self.db_kwargs.get("cached_statements", 128)
        cached_statements = int(raw_cached) if isinstance(raw_cached, int) else 128
        raw_uri = bool(self.db_kwargs.get("uri", False))
        raw_same_thread = self.db_kwargs.get("check_same_thread")
        check_same_thread = (
            False
            if (self.db_path == ":memory:" and raw_same_thread is None)
            else (bool(raw_same_thread) if raw_same_thread is not None else True)
        )

        if self.db_path == ":memory:":
            target_path = self._actual_path
            uri = True
        elif self.read_only:
            target_path, ro_uri = _build_sqlite_readonly_uri(self.db_path)
            uri = raw_uri or ro_uri
        else:
            target_path = self.db_path
            uri = raw_uri

        conn = sqlite3.connect(
            target_path,
            timeout=timeout,
            detect_types=detect_types,
            check_same_thread=check_same_thread,
            cached_statements=cached_statements,
            uri=uri,
        )
        self._harden_sqlite_connection(conn, enforce_query_only=(self.read_only and self.db_path != ":memory:"))
        return conn

    async def connect_async(self) -> AiosqliteConnectionProtocol:
        """Connect asynchronously to SQLite via `aiosqlite`, reusing any cached connection.

        Returns:
            The active `aiosqlite` connection instance.

        Raises:
            DependencyMissingError: If `aiosqlite` is not installed.

        """
        aio_mod = _resolve_aiosqlite(self._aiosqlite_override)
        if aio_mod is None:
            msg = "aiosqlite is required."
            raise DependencyMissingError(msg, dependency_name="aiosqlite", backend="sqlite")
        async with self._async_lock:
            if self._async_conn is not None:
                return self._async_conn
            kwargs = dict(self.db_kwargs)
            if self.db_path == ":memory:":
                kwargs["uri"] = True
                conn: AiosqliteConnectionProtocol = await aio_mod.connect(self._actual_path, **kwargs)
            elif self.read_only:
                ro_target, ro_uri = _build_sqlite_readonly_uri(self.db_path)
                if ro_uri:
                    kwargs["uri"] = True
                conn = await aio_mod.connect(ro_target, **kwargs)
            else:
                conn = await aio_mod.connect(self.db_path, **kwargs)
            self._async_conn = conn
            return conn

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL statements to set up the SQLite schema, temporarily allowing DDL on `:memory:` or writable DBs.

        Args:
            ddl: DDL string containing SQL statements to execute.

        Raises:
            DatabaseExecutionError: If SQLite DDL execution fails.

        """
        with self._lock:
            try:
                if self.read_only and self.db_path != ":memory:":
                    # Schema setup is an explicit administrative step before read-only evaluation queries
                    setup_conn = sqlite3.connect(self.db_path)
                    try:
                        self._harden_sqlite_connection(setup_conn, enforce_query_only=False)
                        with setup_conn:
                            setup_conn.executescript(ddl)
                    finally:
                        setup_conn.close()
                    if self._conn is not None:
                        self._conn.close()
                        self._conn = None
                else:
                    conn_obj = self.conn
                    with conn_obj:
                        conn_obj.executescript(ddl)
            except sqlite3.Error as e:
                raise DatabaseExecutionError(str(e), cause=e) from e

    def execute_query(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a SQLite query synchronously with thread-lock synchronization.

        Args:
            query: SQL query string to execute.
            params: Optional positional query parameters.

        Returns:
            List of result row tuples.

        Raises:
            DatabaseExecutionError: If SQLite query execution fails.

        """
        with self._lock:
            return super().execute_query(query, params)

    async def execute_query_async(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a SQLite query asynchronously and return fetched rows.

        Args:
            query: SQL query string.
            params: Optional sequence of query parameters.

        Returns:
            List of result row tuples.

        Raises:
            DatabaseExecutionError: If SQLite execution fails under `self.error_classes`.
            Error: If a `sqlite3.Error` escapes when `self.error_classes` is narrowed.
            RuntimeError: If an async runtime failure escapes `self.error_classes`.
            ValueError: If an invalid parameter value escapes `self.error_classes`.
            TypeError: If an invalid parameter type escapes `self.error_classes`.
            OSError: If an I/O error escapes `self.error_classes`.

        """
        try:
            async_conn = await self.connect_async()
            try:
                cursor = await async_conn.execute(query, params or ())
                try:
                    desc = cursor.description if isinstance(cursor, _DescribableAsyncCursorProtocol) else None
                    if desc is not None:
                        fetched = await cursor.fetchall()
                        return [tuple(r) for r in fetched]
                    if isinstance(async_conn, _CommittableAsyncConnProtocol) and callable(async_conn.commit):
                        res_c = async_conn.commit()
                        if asyncio.iscoroutine(res_c):
                            await res_c
                    return []
                finally:
                    if isinstance(cursor, _ClosableAsyncCursorProtocol) and callable(cursor.close):
                        res = cursor.close()
                        if asyncio.iscoroutine(res):
                            await res
            except (sqlite3.Error, RuntimeError, ValueError, TypeError, OSError):
                if isinstance(async_conn, _RollbackableAsyncConnProtocol) and callable(async_conn.rollback):
                    res_r = async_conn.rollback()
                    if asyncio.iscoroutine(res_r):
                        await res_r
                raise
        except self.error_classes as e:
            logger.debug("Async Query execution failed: %s", e)
            raise DatabaseExecutionError(str(e), cause=e) from e


__all__ = ["AiosqliteConnectionProtocol", "AiosqliteModuleProtocol", "SQLiteAdapter"]
