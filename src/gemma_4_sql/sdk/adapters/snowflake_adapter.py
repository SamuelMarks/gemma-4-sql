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
        """Connect synchronously."""
        if snowflake is None:
            msg = "snowflake-connector-python is required."
            raise ImportError(msg)
        return snowflake.connector.connect(**self.db_kwargs)

    async def connect_async(self) -> object:
        """Connect asynchronously."""
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

    def execute_with_feedback(self, query: str) -> tuple[bool, list[tuple[object, ...]], str | None]:
        """Execute synchronously with feedback."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            if e.__class__.__name__ in ("Error", "DatabaseError", "DataError", "ProgrammingError", "OperationalError", "IntegrityError", "InternalError", "NotSupportedError"):
                pass
            return (False, [], str(e))
        else:
            if cursor.description is not None:
                return (True, cursor.fetchall(), None)
            return (True, [], None)
        finally:
            if "cursor" in locals():
                cursor.close()

    async def execute_with_feedback_async(self, _query: str) -> tuple[bool, list[tuple[object, ...]], str | None]:
        """Execute asynchronously with feedback."""
        msg = "Async operations not natively supported for db_type: snowflake"
        raise ValueError(msg)

    def execute_query(self, query: str) -> list[tuple[object, ...]]:
        """Execute synchronously."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            if e.__class__.__name__ in ("Error", "DatabaseError", "DataError", "ProgrammingError", "OperationalError", "IntegrityError", "InternalError", "NotSupportedError"):
                pass
            logger.debug("Query execution failed: %s", e)
            return []
        else:
            if cursor.description is not None:
                return cursor.fetchall()
            return []
        finally:
            if "cursor" in locals():
                cursor.close()

    async def execute_query_async(self, _query: str) -> list[tuple[object, ...]]:
        """Execute asynchronously."""
        msg = "Async operations not natively supported for db_type: snowflake"
        raise ValueError(msg)
