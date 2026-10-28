"""Snowflake database adapter supporting cloud warehouse execution."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import types
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.lazy_loader import LazyLoader
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.type_hints import DatabaseConnectionProtocol, JSONPrimitive

from .base import DatabaseAdapter, DBKwargValue

logger = logging.getLogger(__name__)
_SNOWFLAKE_LOADER: LazyLoader[types.ModuleType] = LazyLoader("snowflake.connector")
snowflake = _SNOWFLAKE_LOADER.get_module()


@runtime_checkable
class SnowflakeModuleProtocol(Protocol):
    """Protocol for `snowflake` or `snowflake.connector` module."""

    def connect(self, **kwargs: DBKwargValue) -> DatabaseConnectionProtocol:
        """Open a synchronous Snowflake connection."""
        ...


class SnowflakeAdapter(DatabaseAdapter[DatabaseConnectionProtocol, DatabaseConnectionProtocol]):
    """Adapter for Snowflake cloud data warehouses."""

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Provide the exception classes for Snowflake errors.

        Returns:
            Tuple of Snowflake connector exception types.

        """
        if snowflake is not None:
            with contextlib.suppress(AttributeError):
                cand_err = snowflake.connector.errors.Error
                if isinstance(cand_err, type) and issubclass(cand_err, Exception):
                    return (cand_err,)
        return (RuntimeError, OSError, DatabaseExecutionError)

    def connect(self) -> DatabaseConnectionProtocol:
        """Connect synchronously to Snowflake.

        Returns:
            Active Snowflake connection object.

        Raises:
            DependencyMissingError: If `snowflake-connector-python` is missing or invalid.

        """
        if snowflake is None:
            msg = "snowflake-connector-python is required. Install with `pip install snowflake-connector-python`."
            raise DependencyMissingError(msg, dependency_name="snowflake-connector-python", backend="snowflake")
        connect_fn: Callable[..., DatabaseConnectionProtocol] | None = None
        with contextlib.suppress(AttributeError):
            connect_fn = snowflake.connector.connect
        if not callable(connect_fn):
            with contextlib.suppress(AttributeError):
                connect_fn = snowflake.connect
        if not callable(connect_fn):
            msg = "snowflake connect function could not be resolved."
            raise DependencyMissingError(msg, dependency_name="snowflake-connector-python", backend="snowflake")
        return connect_fn(**self.db_kwargs)

    async def connect_async(self) -> DatabaseConnectionProtocol:
        """Connect asynchronously to Snowflake using a thread worker.

        Returns:
            The established Snowflake connection object.

        """
        conn = await asyncio.to_thread(self.connect)
        self.conn = conn
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
            with contextlib.suppress(AttributeError):
                conn_obj.commit()
        except self.error_classes + (RuntimeError, ValueError, TypeError, OSError) as e:
            with contextlib.suppress(AttributeError):
                conn_obj.rollback()
            raise DatabaseExecutionError(str(e), cause=e) from e
        finally:
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
