"""Snowflake adapter."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONPrimitive

from gemma_4_sql.backends.lazy_loader import LazyLoader

from .base import DatabaseAdapter

logger = logging.getLogger(__name__)
snowflake = LazyLoader("snowflake.connector").get_module()


class SnowflakeAdapter(DatabaseAdapter):
    """Adapter for Snowflake."""

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Return the exception classes."""
        try:
            import snowflake.connector

            err_cls = getattr(getattr(snowflake.connector, "errors", None), "Error", None)
            if isinstance(err_cls, type) and issubclass(err_cls, Exception):
                return (err_cls,)
            return (Exception,)
        except (ImportError, AttributeError):
            return (Exception,)

    def connect(self) -> object:
        """Connect synchronously.

        Returns:
            The execution result.

        Raises:
            ImportError: If snowflake-connector-python is missing.
        """
        if snowflake is None:
            msg = "snowflake-connector-python is required. Install with `pip install snowflake-connector-python`."
            raise ImportError(msg)
        if hasattr(snowflake, "connector") and hasattr(snowflake.connector, "connect"):
            connect_fn = snowflake.connector.connect
        else:
            connect_fn = getattr(snowflake, "connect", None)
        if connect_fn is None:
            msg = "snowflake connect function could not be resolved."
            raise ImportError(msg)
        return connect_fn(**self.db_kwargs)

    async def connect_async(self) -> object:
        """Connect asynchronously using a thread worker.

        Returns:
            The established connection object.
        """
        conn = await asyncio.to_thread(self.connect)
        self.conn = conn
        return conn

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL to set up schema.

        Args:
            ddl: The DDL string to execute.
        """
        conn_obj = cast(Any, self.conn)
        cursor = conn_obj.cursor()
        try:
            cursor.execute(ddl)
            if hasattr(conn_obj, "commit"):
                conn_obj.commit()
        except Exception:
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
        """Execute asynchronously with feedback via thread offload.

        Args:
            query: The SQL query string to execute.
            params: Optional query parameters.

        Returns:
            Tuple of success status, result rows, and error message if failed.
        """
        return await asyncio.to_thread(self.execute_with_feedback, query, params)

    async def execute_query_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute query asynchronously via thread offload.

        Args:
            query: The SQL query string to execute.
            params: Optional query parameters.

        Returns:
            List of query result tuples.
        """
        return await asyncio.to_thread(self.execute_query, query, params)
