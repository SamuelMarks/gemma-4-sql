"""Base adapter protocol for database connections."""

from __future__ import annotations

import abc
import asyncio
import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.exceptions import DatabaseExecutionError

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONPrimitive

logger = logging.getLogger(__name__)


class DatabaseAdapter(abc.ABC):
    """Abstract base class for database dialect adapters."""

    def __init__(self, db_path: str, db_kwargs: dict[str, object], *, read_only: bool = False) -> None:
        """Initialize the database adapter and establish initial connection.

        Args:
            db_path: The file path or connection URI for the database.
            db_kwargs: Additional connection keyword arguments.
            read_only: Flag indicating whether mutating queries are disallowed.

        """
        self.db_path = db_path
        self.db_kwargs = db_kwargs
        self.read_only = read_only
        self.conn: Any = self.connect()

    @property
    @abc.abstractmethod
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Provide the tuple of database-specific exception types caught by this adapter."""

    @abc.abstractmethod
    def connect(self) -> object:
        """Connect synchronously to the underlying database.

        Returns:
            The active database connection object.

        """

    async def connect_async(self) -> object:
        """Connect asynchronously to the underlying database.

        Returns:
            The active asynchronous database connection object.

        """
        return await asyncio.to_thread(self.connect)

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL statements to initialize the database schema.

        Args:
            ddl: The SQL DDL query string.

        """
        self.execute_with_feedback(ddl)

    def execute_with_feedback(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute a query synchronously and return execution status, rows, and error feedback.

        Args:
            query: The SQL query string to execute.
            params: Optional positional parameters for the query.

        Returns:
            A tuple containing `(success, rows, error_message)`.

        """
        try:
            rows = self.execute_query(query, params)
        except self.error_classes + (DatabaseExecutionError,) as e:
            cause = e.cause if isinstance(e, DatabaseExecutionError) and e.cause is not None else e
            return (False, [], str(cause))
        else:
            return (True, rows, None)

    async def execute_with_feedback_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute a query asynchronously and return execution status, rows, and error feedback.

        Args:
            query: The SQL query string to execute.
            params: Optional positional parameters for the query.

        Returns:
            A tuple containing `(success, rows, error_message)`.

        """
        return await asyncio.to_thread(self.execute_with_feedback, query, params)

    def execute_query(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a query synchronously and return fetched result tuples.

        Args:
            query: The SQL query string to execute.
            params: Optional positional parameters for the query.

        Returns:
            List of result row tuples.

        Raises:
            DatabaseExecutionError: If query execution fails on the underlying connection.

        """
        try:
            if hasattr(self.conn, "cursor"):
                cursor = self.conn.cursor()
                cursor.execute(query, params or ())
                results: list[tuple[JSONPrimitive, ...]] = cursor.fetchall() if getattr(cursor, "description", None) is not None else []
                if hasattr(cursor, "close"):
                    cursor.close()
                return results
            return self.conn.execute(query, params or ()).fetchall()
        except self.error_classes as e:
            logger.debug("Query execution failed: %s", e)
            raise DatabaseExecutionError(str(e), cause=e) from e

    async def execute_query_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a query asynchronously and return fetched result tuples.

        Args:
            query: The SQL query string to execute.
            params: Optional positional parameters for the query.

        Returns:
            List of result row tuples.

        Raises:
            DatabaseExecutionError: If query execution fails on the underlying connection.

        """
        return await asyncio.to_thread(self.execute_query, query, params)

    def close(self) -> None:
        """Close the active database connection if open."""
        if hasattr(self.conn, "close"):
            self.conn.close()
