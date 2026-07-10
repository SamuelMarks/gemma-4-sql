"""Snowflake adapter."""

from __future__ import annotations

import logging

from gemma_4_sql.backends.lazy_loader import LazyLoader

from .base import DatabaseAdapter

logger = logging.getLogger(__name__)
snowflake = LazyLoader("snowflake.connector").get_module()


class SnowflakeAdapter(DatabaseAdapter):
    """Adapter for Snowflake."""

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Return the exception classes."""
        import snowflake.connector

        return (snowflake.connector.errors.Error,)

    def connect(self) -> object:
        """Connect synchronously.

        Returns:
            The execution result.
        """
        if snowflake is None:
            msg = "snowflake-connector-python is required."
            raise ImportError(msg)
        return snowflake.connector.connect(**self.db_kwargs)

    async def connect_async(self) -> object:
        """Connect asynchronously.

        Raises:
        ValueError: If the operation encounters an unexpected ValueError.

        """
        msg = "Async operations not natively supported for db_type: snowflake"
        raise ValueError(msg)

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL to set up schema."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(ddl)
            self.conn.commit()
        finally:
            cursor.close()

    async def execute_with_feedback_async(self, _query: str, params: tuple[object, ...] | None = None) -> tuple[bool, list[tuple[object, ...]], str | None]:
        """Execute asynchronously with feedback.

        Raises:
        ValueError: If the operation encounters an unexpected ValueError.

        """
        msg = "Async operations not natively supported for db_type: snowflake"
        raise ValueError(msg)

    async def execute_query_async(self, _query: str, params: tuple[object, ...] | None = None) -> list[tuple[object, ...]]:
        """Execute asynchronously.

        Raises:
        ValueError: If the operation encounters an unexpected ValueError.

        """
        msg = "Async operations not natively supported for db_type: snowflake"
        raise ValueError(msg)
