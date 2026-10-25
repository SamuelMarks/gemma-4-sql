"""Snowflake database adapter supporting cloud warehouse execution."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import LazyLoader
from gemma_4_sql.exceptions import DependencyMissingError

from .base import DatabaseAdapter

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONPrimitive

logger = logging.getLogger(__name__)
snowflake = LazyLoader("snowflake.connector").get_module()


class SnowflakeAdapter(DatabaseAdapter):
    """Adapter for Snowflake cloud data warehouses."""

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Provide the exception classes for Snowflake errors.

        Returns:
            Tuple of Snowflake connector exception types.

        """
        sf_pkg = LazyLoader("snowflake.connector").get_module()
        connector_any: object = getattr(snowflake, "connector", sf_pkg)
        err_cls = getattr(getattr(connector_any, "errors", None), "Error", None)
        if isinstance(err_cls, type) and issubclass(err_cls, Exception):
            return (err_cls,)
        return (Exception,)

    def connect(self) -> object:
        """Connect synchronously to Snowflake.

        Returns:
            Active Snowflake connection object.

        Raises:
            DependencyMissingError: If `snowflake-connector-python` is missing or invalid.

        """
        if snowflake is None:
            msg = "snowflake-connector-python is required. Install with `pip install snowflake-connector-python`."
            raise DependencyMissingError(msg, dependency_name="snowflake-connector-python", backend="snowflake")
        if hasattr(snowflake, "connector") and hasattr(snowflake.connector, "connect"):
            connect_fn = snowflake.connector.connect
        else:
            connect_fn = getattr(snowflake, "connect", None)
        if connect_fn is None:
            msg = "snowflake connect function could not be resolved."
            raise DependencyMissingError(msg, dependency_name="snowflake-connector-python", backend="snowflake")
        return connect_fn(**self.db_kwargs)

    async def connect_async(self) -> object:
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

        """
        conn_obj = self.conn
        cursor = conn_obj.cursor()
        try:
            cursor.execute(ddl)
            if hasattr(conn_obj, "commit"):
                conn_obj.commit()
        except self.error_classes + (RuntimeError, ValueError, TypeError, OSError):
            if hasattr(conn_obj, "rollback"):
                conn_obj.rollback()
            raise
        finally:
            cursor.close()

    async def execute_with_feedback_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute a Snowflake query asynchronously with feedback via thread offload.

        Args:
            query: The SQL query string to execute.
            params: Optional query parameters.

        Returns:
            Tuple of `(success, rows, error_message)`.

        """
        return await asyncio.to_thread(self.execute_with_feedback, query, params)

    async def execute_query_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
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
