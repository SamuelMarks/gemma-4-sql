# Copyright 2024
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
            object: The resulting output from the operation.

        Raises:
        ImportError: If the operation encounters an unexpected ImportError.

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

    def execute_with_feedback(self, query: str, params: tuple[object, ...] | None = None) -> tuple[bool, list[tuple[object, ...]], str | None]:
        """Execute synchronously with feedback.

        Returns:
            object: The resulting output from the operation.

        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
        except Exception as e:  # noqa: BLE001
            return (False, [], str(e))
        else:
            if cursor.description is not None:
                return (True, cursor.fetchall(), None)
            return (True, [], None)
        finally:
            if "cursor" in locals():
                cursor.close()

    async def execute_with_feedback_async(self, _query: str, params: tuple[object, ...] | None = None) -> tuple[bool, list[tuple[object, ...]], str | None]:
        """Execute asynchronously with feedback.

        Raises:
        ValueError: If the operation encounters an unexpected ValueError.

        """
        msg = "Async operations not natively supported for db_type: snowflake"
        raise ValueError(msg)

    def execute_query(self, query: str, params: tuple[object, ...] | None = None) -> list[tuple[object, ...]]:
        """Execute synchronously.

        Returns:
            object: The resulting output from the operation.

        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
        except Exception as e:  # noqa: BLE001
            logger.debug("Query execution failed: %s", e)
            return []
        else:
            if cursor.description is not None:
                return cursor.fetchall()
            return []
        finally:
            if "cursor" in locals():
                cursor.close()

    async def execute_query_async(self, _query: str, params: tuple[object, ...] | None = None) -> list[tuple[object, ...]]:
        """Execute asynchronously.

        Raises:
        ValueError: If the operation encounters an unexpected ValueError.

        """
        msg = "Async operations not natively supported for db_type: snowflake"
        raise ValueError(msg)
