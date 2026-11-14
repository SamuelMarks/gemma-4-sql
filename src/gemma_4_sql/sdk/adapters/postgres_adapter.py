"""PostgreSQL database adapter supporting psycopg2 and asyncpg with placeholder normalization."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Coroutine, Sequence
from enum import Enum
from typing import Protocol, Self, runtime_checkable

from gemma_4_sql.backends.lazy_loader import LazyLoader, ModuleNamespaceProtocol
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.sdk.sql_lexer import scan_sql_segments
from gemma_4_sql.type_hints import (
    AsyncDatabaseConnectionProtocol,
    JSONPrimitive,
)

from .base import CursorFactoryConnectionProtocol, DatabaseAdapter, DBKwargValue

logger = logging.getLogger(__name__)
_DOLLAR_PARAM_PATTERN = re.compile(r"\$([1-9][0-9]*)\b")


class _LoaderSentinel(Enum):
    """Sentinel indicating a PostgreSQL driver has not yet been lazily imported."""

    UNRESOLVED = "unresolved"


@runtime_checkable
class _AsyncpgRecordProtocol(Protocol):
    """Protocol for an `asyncpg.Record` exposing `.values()`."""

    def values(self) -> Sequence[JSONPrimitive]:
        """Return record column values.

        Returns:
            Sequence of column values in the row.

        """
        ...


@runtime_checkable
class _AsyncpgExecutableProtocol(Protocol):
    """Protocol for `asyncpg` connections supporting `.execute(...)` session commands."""

    async def execute(self, query: str) -> str | None:
        """Execute a session command such as `SET DEFAULT_TRANSACTION_READ_ONLY = ON`.

        Args:
            query: SQL statement text.

        Returns:
            Optional status tag string.

        """
        ...


@runtime_checkable
class _AsyncpgConnectionProtocol(AsyncDatabaseConnectionProtocol, Protocol):
    """Protocol for an `asyncpg.Connection` supporting `.fetch(...)` and `.close()`."""

    async def fetch(self, query: str, *args: JSONPrimitive) -> Sequence[_AsyncpgRecordProtocol]:
        """Execute a SQL query and return fetched records.

        Args:
            query: SQL query text using `$1, $2` positional placeholders.
            *args: Positional parameter values.

        Returns:
            Sequence of `_AsyncpgRecordProtocol` rows.

        """
        ...


@runtime_checkable
class _ClosableAsyncProtocol(Protocol):
    """Protocol for async connections exposing `.close()`."""

    def close(self) -> Coroutine[None, None, None] | None:
        """Close the async connection."""
        ...


@runtime_checkable
class _TransactableConnProtocol(Protocol):
    """Protocol for synchronous connections exposing `.commit()` and `.rollback()`."""

    def commit(self) -> Self | None:
        """Commit the active transaction."""
        ...

    def rollback(self) -> Self | None:
        """Roll back the active transaction."""
        ...


@runtime_checkable
class _Psycopg2ErrorProtocol(Protocol):
    """Protocol for `psycopg2` exposing `.Error`."""

    @property
    def Error(self) -> type[BaseException] | int | str | None:
        """Provide `psycopg2.Error` exception class."""
        ...


@runtime_checkable
class Psycopg2ModuleProtocol(Protocol):
    """Protocol for the `psycopg2` Python driver module."""

    def connect(self, dsn: str = "", /, **kwargs: DBKwargValue) -> CursorFactoryConnectionProtocol:
        """Open a synchronous `psycopg2` connection.

        Args:
            dsn: PostgreSQL connection string.
            **kwargs: Driver connection keyword arguments.

        Returns:
            Connected `CursorFactoryConnectionProtocol` instance.

        """
        ...


@runtime_checkable
class _AsyncpgErrorProtocol(Protocol):
    """Protocol for `asyncpg` exposing `.PostgresError`."""

    @property
    def PostgresError(self) -> type[BaseException] | int | str | None:
        """Provide `asyncpg.PostgresError` exception class."""
        ...


@runtime_checkable
class AsyncpgModuleProtocol(Protocol):
    """Protocol for the `asyncpg` Python driver module."""

    async def connect(self, dsn: str = "", /, **kwargs: DBKwargValue) -> _AsyncpgConnectionProtocol:
        """Open an asynchronous `asyncpg` connection.

        Args:
            dsn: PostgreSQL connection string.
            **kwargs: Driver connection keyword arguments.

        Returns:
            Connected `_AsyncpgConnectionProtocol` instance.

        """
        ...


_PSYCOPG2_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("psycopg2")
_ASYNCPG_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("asyncpg")
psycopg2: Psycopg2ModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None = _LoaderSentinel.UNRESOLVED
asyncpg: AsyncpgModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None = _LoaderSentinel.UNRESOLVED


def _resolve_psycopg2(
    override: Psycopg2ModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None = _LoaderSentinel.UNRESOLVED,
) -> Psycopg2ModuleProtocol | ModuleNamespaceProtocol | None:
    """Resolve `psycopg2` lazily while honoring explicit DI or module-attribute overrides.

    Args:
        override: Optional explicitly injected `psycopg2` module or `None`.

    Returns:
        Loaded `psycopg2` module or `None`.

    """
    global psycopg2
    if override is not _LoaderSentinel.UNRESOLVED:
        return override
    if psycopg2 is _LoaderSentinel.UNRESOLVED:
        psycopg2 = _PSYCOPG2_LOADER.get_module()
    return psycopg2


def _resolve_asyncpg(
    override: AsyncpgModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None = _LoaderSentinel.UNRESOLVED,
) -> AsyncpgModuleProtocol | ModuleNamespaceProtocol | None:
    """Resolve `asyncpg` lazily while honoring explicit DI or module-attribute overrides.

    Args:
        override: Optional explicitly injected `asyncpg` module or `None`.

    Returns:
        Loaded `asyncpg` module or `None`.

    """
    global asyncpg
    if override is not _LoaderSentinel.UNRESOLVED:
        return override
    if asyncpg is _LoaderSentinel.UNRESOLVED:
        asyncpg = _ASYNCPG_LOADER.get_module()
    return asyncpg


def normalize_to_asyncpg_placeholders(
    query: str, params: Sequence[JSONPrimitive]
) -> tuple[str, tuple[JSONPrimitive, ...]]:
    """Rewrite unquoted `%s` placeholders into `$1, $2, ...` for `asyncpg` while preserving literals/comments.

    Args:
        query: SQL query using either `%s` or `$1` placeholders.
        params: Positional query parameters.

    Returns:
        Tuple of `(normalized_query, ordered_params)`.

    """
    if not params:
        return (query, ())
    idx = 0
    out_parts: list[str] = []
    for seg in scan_sql_segments(query):
        if seg.kind == "code" and "%s" in seg.text:
            pieces = seg.text.split("%s")
            rebuilt = [pieces[0]]
            for piece in pieces[1:]:
                idx += 1
                rebuilt.extend((f"${idx}", piece))
            out_parts.append("".join(rebuilt))
        else:
            out_parts.append(seg.text)
    return ("".join(out_parts), tuple(params))


def normalize_to_psycopg2_placeholders(
    query: str, params: Sequence[JSONPrimitive]
) -> tuple[str, tuple[JSONPrimitive, ...]]:
    """Rewrite unquoted `$1, $2, ...` placeholders into `%s` for `psycopg2` while preserving literals/comments.

    Args:
        query: SQL query using either `%s` or `$1` placeholders.
        params: Positional query parameters.

    Returns:
        Tuple of `(normalized_query, expanded_params)`.

    """
    if not params:
        return (query, ())
    segments = scan_sql_segments(query)
    has_dollar = any(seg.kind == "code" and _DOLLAR_PARAM_PATTERN.search(seg.text) is not None for seg in segments)
    if not has_dollar:
        return (query, tuple(params))

    expanded: list[JSONPrimitive] = []
    out_parts: list[str] = []

    def _replace_match(m: re.Match[str]) -> str:
        """Map `$k` 1-based positional placeholder to `%s` and append `params[k - 1]`.

        Args:
            m: Regex match for `$k`.

        Returns:
            Replacement `%s` token or original text if index exceeds `params`.

        """
        pos = int(m.group(1)) - 1
        if 0 <= pos < len(params):
            expanded.append(params[pos])
            return "%s"
        return m.group(0)

    for seg in segments:
        if seg.kind == "code":
            out_parts.append(_DOLLAR_PARAM_PATTERN.sub(_replace_match, seg.text))
        else:
            out_parts.append(seg.text)
    return ("".join(out_parts), tuple(expanded))


class PostgresAdapter(DatabaseAdapter[CursorFactoryConnectionProtocol, _AsyncpgConnectionProtocol]):
    """Adapter for PostgreSQL databases using `psycopg2` and `asyncpg`."""

    _psycopg2_override: Psycopg2ModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None
    _asyncpg_override: AsyncpgModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None

    def __init__(
        self,
        db_path: str = "",
        db_kwargs: dict[str, DBKwargValue] | None = None,
        *,
        read_only: bool = True,
        lazy_connect: bool = False,
        psycopg2_module: (
            Psycopg2ModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None
        ) = _LoaderSentinel.UNRESOLVED,
        asyncpg_module: (
            AsyncpgModuleProtocol | ModuleNamespaceProtocol | _LoaderSentinel | None
        ) = _LoaderSentinel.UNRESOLVED,
    ) -> None:
        """Initialize the PostgreSQL database adapter.

        Args:
            db_path: Connection DSN or URI string.
            db_kwargs: Optional connection keyword arguments.
            read_only: Whether to enforce read-only query execution.
            lazy_connect: Whether to defer synchronous `psycopg2` connection until `.conn` access.
            psycopg2_module: Optional injected `psycopg2` driver module or `None` for testing.
            asyncpg_module: Optional injected `asyncpg` driver module or `None` for testing.

        """
        self._psycopg2_override = psycopg2_module
        self._asyncpg_override = asyncpg_module
        super().__init__(db_path, db_kwargs or {}, read_only=read_only, lazy_connect=lazy_connect)

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Provide the exception classes for PostgreSQL errors.

        Returns:
            Tuple of PostgreSQL driver exception types.

        """
        classes: list[type[Exception]] = []
        pg_mod = _resolve_psycopg2(self._psycopg2_override)
        if isinstance(pg_mod, _Psycopg2ErrorProtocol):
            pg_err = pg_mod.Error
            if isinstance(pg_err, type) and issubclass(pg_err, Exception):
                classes.append(pg_err)
        apg_mod = _resolve_asyncpg(self._asyncpg_override)
        if isinstance(apg_mod, _AsyncpgErrorProtocol):
            apg_err = apg_mod.PostgresError
            if isinstance(apg_err, type) and issubclass(apg_err, Exception):
                classes.append(apg_err)
        if not classes:
            return (RuntimeError, OSError, DatabaseExecutionError)
        return tuple(classes)

    def connect(self) -> CursorFactoryConnectionProtocol:
        """Connect synchronously to PostgreSQL via `psycopg2`.

        Returns:
            Active `psycopg2` connection object.

        Raises:
            DependencyMissingError: If `psycopg2` is not installed.

        """
        pg_mod = _resolve_psycopg2(self._psycopg2_override)
        if not isinstance(pg_mod, Psycopg2ModuleProtocol):
            msg = "psycopg2 is required. Install with `pip install psycopg2-binary`."
            raise DependencyMissingError(msg, dependency_name="psycopg2", backend="postgresql")
        kw = dict(self.db_kwargs)
        explicit_dsn = kw.pop("dsn", None)
        dsn = (
            str(explicit_dsn)
            if explicit_dsn is not None
            else (self.db_path if self.db_path and self.db_path != ":memory:" else "")
        )
        if dsn:
            conn: CursorFactoryConnectionProtocol = pg_mod.connect(dsn, **kw)
        else:
            conn = pg_mod.connect(**kw)
        if self.read_only:
            cur = conn.cursor()
            try:
                cur.execute("SET DEFAULT_TRANSACTION_READ_ONLY = ON")
            except self.error_classes:
                pass
            finally:
                if isinstance(cur, _ClosableAsyncProtocol) and callable(cur.close):
                    cur.close()
        return conn

    async def connect_async(self) -> _AsyncpgConnectionProtocol:
        """Connect asynchronously to PostgreSQL via `asyncpg`, reusing any cached connection.

        Returns:
            Active `asyncpg` connection object.

        Raises:
            DependencyMissingError: If `asyncpg` is not installed.

        """
        apg_mod = _resolve_asyncpg(self._asyncpg_override)
        if not isinstance(apg_mod, AsyncpgModuleProtocol):
            msg = "asyncpg is required."
            raise DependencyMissingError(msg, dependency_name="asyncpg", backend="postgresql")
        async with self._async_lock:
            if self._async_conn is not None:
                return self._async_conn
            kw = dict(self.db_kwargs)
            explicit_dsn = kw.pop("dsn", None)
            dsn = (
                str(explicit_dsn)
                if explicit_dsn is not None
                else (self.db_path if self.db_path and self.db_path != ":memory:" else "")
            )
            if dsn:
                conn: _AsyncpgConnectionProtocol = await apg_mod.connect(dsn, **kw)
            else:
                conn = await apg_mod.connect(**kw)
            if self.read_only and isinstance(conn, _AsyncpgExecutableProtocol) and callable(conn.execute):
                try:
                    await conn.execute("SET DEFAULT_TRANSACTION_READ_ONLY = ON")
                except self.error_classes:
                    pass
            self._async_conn = conn
            return conn

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL statements to initialize the PostgreSQL schema.

        Args:
            ddl: DDL SQL statements string.

        Raises:
            DatabaseExecutionError: If PostgreSQL DDL execution fails.

        """
        conn_obj = self.conn
        cursor = conn_obj.cursor()
        try:
            cursor.execute(ddl)
            if isinstance(conn_obj, _TransactableConnProtocol) and callable(conn_obj.commit):
                conn_obj.commit()
        except (*self.error_classes, RuntimeError, ValueError, TypeError, OSError) as e:
            if isinstance(conn_obj, _TransactableConnProtocol) and callable(conn_obj.rollback):
                conn_obj.rollback()
            raise DatabaseExecutionError(str(e), cause=e) from e
        finally:
            if isinstance(cursor, _ClosableAsyncProtocol) and callable(cursor.close):
                cursor.close()

    def execute_query(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a PostgreSQL query synchronously, normalizing `$1` or `%s` placeholders.

        Args:
            query: The SQL query string.
            params: Optional positional query parameters.

        Returns:
            List of result row tuples.

        Raises:
            DatabaseExecutionError: If PostgreSQL query execution fails.

        """
        norm_query, norm_params = normalize_to_psycopg2_placeholders(query, params or ())
        return super().execute_query(norm_query, norm_params if params is not None else None)

    async def execute_query_async(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a PostgreSQL query asynchronously via `asyncpg`, normalizing `%s` or `$1` placeholders.

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
            norm_query, query_params = normalize_to_asyncpg_placeholders(query, params or ())
            records = await async_conn.fetch(norm_query, *query_params)
            return [tuple(r.values()) for r in records]
        except self.error_classes as e:
            failed_conn = self._async_conn
            self._async_conn = None
            if failed_conn is not None and callable(failed_conn.close):
                try:
                    close_res = failed_conn.close()
                    if asyncio.iscoroutine(close_res):
                        await close_res
                except (RuntimeError, OSError):
                    pass
            logger.debug("Async Query execution failed: %s", e)
            raise DatabaseExecutionError(str(e), cause=e) from e


__all__ = [
    "AsyncpgModuleProtocol",
    "PostgresAdapter",
    "Psycopg2ModuleProtocol",
    "normalize_to_asyncpg_placeholders",
    "normalize_to_psycopg2_placeholders",
]
