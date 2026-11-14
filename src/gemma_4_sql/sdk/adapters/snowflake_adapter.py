"""Snowflake database adapter supporting cloud warehouse execution."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Protocol, Self, runtime_checkable

from gemma_4_sql.backends.lazy_loader import LazyLoader, ModuleNamespaceProtocol
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.type_hints import JSONPrimitive

from .base import ClosableResourceProtocol, CursorFactoryConnectionProtocol, DatabaseAdapter, DBKwargValue

logger = logging.getLogger(__name__)


class _LoaderSentinel(Enum):
    """Sentinel indicating `snowflake.connector` has not yet been lazily imported."""

    UNRESOLVED = "unresolved"


@runtime_checkable
class _SnowflakeErrorsNamespaceProtocol(Protocol):
    """Protocol for `snowflake.connector.errors` exposing `Error`."""

    @property
    def Error(self) -> type[BaseException] | int | str | None:
        """Provide `snowflake.connector.errors.Error` exception class."""
        ...


@runtime_checkable
class _SnowflakeConnectorSubmoduleProtocol(Protocol):
    """Protocol for `snowflake.connector` submodule exposing `connect` and `errors`."""

    def connect(self, **kwargs: DBKwargValue) -> CursorFactoryConnectionProtocol:
        """Open a synchronous Snowflake connection."""
        ...


@runtime_checkable
class _SnowflakeConnectorErrorsHolderProtocol(Protocol):
    """Protocol for `snowflake.connector` exposing `.errors`."""

    @property
    def errors(self) -> _SnowflakeErrorsNamespaceProtocol:
        """Provide `snowflake.connector.errors` namespace."""
        ...


@runtime_checkable
class _SnowflakeRootConnectorHolderProtocol(Protocol):
    """Protocol for top-level `snowflake` package exposing `.connector`."""

    @property
    def connector(self) -> _SnowflakeConnectorSubmoduleProtocol | _SnowflakeConnectorErrorsHolderProtocol:
        """Provide `snowflake.connector` submodule."""
        ...


@runtime_checkable
class SnowflakeModuleProtocol(Protocol):
    """Protocol for `snowflake` or `snowflake.connector` module exposing `connect`."""

    def connect(self, **kwargs: DBKwargValue) -> CursorFactoryConnectionProtocol:
        """Open a synchronous Snowflake connection.

        Args:
            **kwargs: Snowflake connection parameters.

        Returns:
            Active `CursorFactoryConnectionProtocol` connection.

        """
        ...


@runtime_checkable
class _TransactableSnowflakeConnProtocol(Protocol):
    """Protocol for Snowflake connections exposing `.commit()` and `.rollback()`."""

    def commit(self) -> Self | None:
        """Commit the active transaction."""
        ...

    def rollback(self) -> Self | None:
        """Roll back the active transaction."""
        ...


type SnowflakeDriverModule = SnowflakeModuleProtocol | _SnowflakeRootConnectorHolderProtocol | ModuleNamespaceProtocol

_SNOWFLAKE_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("snowflake.connector")
snowflake: SnowflakeDriverModule | _LoaderSentinel | None = _LoaderSentinel.UNRESOLVED


def _resolve_snowflake(
    override: SnowflakeDriverModule | _LoaderSentinel | None = _LoaderSentinel.UNRESOLVED,
) -> SnowflakeDriverModule | None:
    """Resolve `snowflake.connector` lazily while honoring explicit DI or module-attribute overrides.

    Args:
        override: Optional explicitly injected Snowflake driver module or `None`.

    Returns:
        Loaded Snowflake module/package or `None`.

    """
    global snowflake
    if override is not _LoaderSentinel.UNRESOLVED:
        return override
    if snowflake is _LoaderSentinel.UNRESOLVED:
        snowflake = _SNOWFLAKE_LOADER.get_module()
    return snowflake


class SnowflakeAdapter(DatabaseAdapter[CursorFactoryConnectionProtocol, CursorFactoryConnectionProtocol]):
    """Adapter for Snowflake cloud data warehouses."""

    _snowflake_override: SnowflakeDriverModule | _LoaderSentinel | None

    def __init__(
        self,
        db_path: str = "",
        db_kwargs: dict[str, DBKwargValue] | None = None,
        *,
        read_only: bool = True,
        lazy_connect: bool = False,
        snowflake_module: SnowflakeDriverModule | _LoaderSentinel | None = _LoaderSentinel.UNRESOLVED,
    ) -> None:
        """Initialize the Snowflake database adapter.

        Args:
            db_path: Optional Snowflake account or warehouse identifier.
            db_kwargs: Optional Snowflake connection keyword arguments.
            read_only: Whether to enforce read-only query validation.
            lazy_connect: Whether to defer synchronous connection establishment until `.conn` access.
            snowflake_module: Optional injected Snowflake driver module/package or `None` for testing.

        """
        self._snowflake_override = snowflake_module
        super().__init__(db_path, db_kwargs or {}, read_only=read_only, lazy_connect=lazy_connect)

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Provide the exception classes for Snowflake errors.

        Returns:
            Tuple of Snowflake connector exception types.

        """
        sf_mod = _resolve_snowflake(self._snowflake_override)
        if isinstance(sf_mod, _SnowflakeRootConnectorHolderProtocol):
            sub = sf_mod.connector
            if isinstance(sub, _SnowflakeConnectorErrorsHolderProtocol):
                err_attr = sub.errors.Error
                if isinstance(err_attr, type) and issubclass(err_attr, Exception):
                    return (err_attr,)
        return (RuntimeError, OSError, DatabaseExecutionError)

    def connect(self) -> CursorFactoryConnectionProtocol:
        """Connect synchronously to Snowflake.

        Returns:
            Active Snowflake connection object.

        Raises:
            DependencyMissingError: If `snowflake-connector-python` is missing or invalid.

        """
        sf_mod = _resolve_snowflake(self._snowflake_override)
        if sf_mod is None:
            msg = "snowflake-connector-python is required. Install with `pip install snowflake-connector-python`."
            raise DependencyMissingError(msg, dependency_name="snowflake-connector-python", backend="snowflake")
        connect_fn: Callable[..., CursorFactoryConnectionProtocol] | None = None
        if (
            isinstance(sf_mod, _SnowflakeRootConnectorHolderProtocol)
            and isinstance(sf_mod.connector, _SnowflakeConnectorSubmoduleProtocol)
            and callable(sf_mod.connector.connect)
        ):
            connect_fn = sf_mod.connector.connect
        elif isinstance(sf_mod, SnowflakeModuleProtocol) and callable(sf_mod.connect):
            connect_fn = sf_mod.connect
        if connect_fn is None:
            msg = "snowflake connect function could not be resolved."
            raise DependencyMissingError(msg, dependency_name="snowflake-connector-python", backend="snowflake")
        return connect_fn(**self.db_kwargs)

    async def connect_async(self) -> CursorFactoryConnectionProtocol:
        """Return the active Snowflake connection or connect lazily via a worker thread without leaking.

        Returns:
            The established Snowflake connection object.

        """
        async with self._async_lock:
            if self._conn is not None:
                return self._conn
            conn = await asyncio.to_thread(self.connect)
            self._conn = conn
            return conn

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL statements to initialize the Snowflake schema.

        Args:
            ddl: The DDL string to execute.

        Raises:
            DatabaseExecutionError: If Snowflake DDL execution fails.

        """
        conn_obj = self.conn
        cursor = conn_obj.cursor()
        try:
            cursor.execute(ddl)
            if isinstance(conn_obj, _TransactableSnowflakeConnProtocol) and callable(conn_obj.commit):
                conn_obj.commit()
        except (*self.error_classes, RuntimeError, ValueError, TypeError, OSError) as e:
            if isinstance(conn_obj, _TransactableSnowflakeConnProtocol) and callable(conn_obj.rollback):
                conn_obj.rollback()
            raise DatabaseExecutionError(str(e), cause=e) from e
        finally:
            if isinstance(cursor, ClosableResourceProtocol) and callable(cursor.close):
                cursor.close()

    async def execute_query_async(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a Snowflake query asynchronously via thread offload.

        Args:
            query: The SQL query string to execute.
            params: Optional query parameters.

        Returns:
            List of query result row tuples.

        Raises:
            DatabaseExecutionError: If Snowflake query execution fails.

        """
        return await asyncio.to_thread(self.execute_query, query, params)


__all__ = ["SnowflakeAdapter", "SnowflakeModuleProtocol"]
