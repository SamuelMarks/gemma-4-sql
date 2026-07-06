"""Snowflake adapter."""

from __future__ import annotations

import logging

from gemma_4_sql.backends.lazy_loader import LazyLoader

from .base import DatabaseAdapter

logger = logging.getLogger(__name__)
snowflake = LazyLoader("snowflake.connector").get_module()


class SnowflakeAdapter(DatabaseAdapter):
    """Adapter for Snowflake."""

    def connect(self) -> object:
        """Connect synchronously.

        Returns:
            The execution result.
        """
        if snowflake is None:
            msg = "snowflake-connector-python is required."  # pragma: no cover
            raise ImportError(msg)  # pragma: no cover
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
        cursor = self.conn.cursor()  # pragma: no cover
        try:  # pragma: no cover
            cursor.execute(ddl)  # pragma: no cover
            self.conn.commit()  # pragma: no cover
        finally:  # pragma: no cover
            cursor.close()  # pragma: no cover

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
