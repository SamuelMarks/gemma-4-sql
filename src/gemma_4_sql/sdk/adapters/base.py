"""Base adapter protocol for database connections."""

from __future__ import annotations

import abc
import logging
import typing

if typing.TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONPrimitive

logger = logging.getLogger(__name__)


class DatabaseAdapter(abc.ABC):
    """Protocol for database dialect adapters."""

    def __init__(self, db_path: str, db_kwargs: dict[str, object], *, read_only: bool = False) -> None:
        """Initialize adapter.

        Args:
            db_path: The file path to the database.
            db_kwargs: A mapping representing db kwargs.
            read_only: Boolean flag indicating read only.
        """
        self.db_path = db_path
        self.db_kwargs = db_kwargs
        self.read_only = read_only
        self.conn = self.connect()

    @abc.abstractmethod
    def connect(self) -> object:
        """Connect synchronously.

        Returns:
            The connection object.
        """

    async def connect_async(self) -> object:
        """Connect asynchronously.

        Returns:
            The async connection object.
        """
        import asyncio

        return await asyncio.to_thread(self.connect)

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL to set up schema.

        Args:
            ddl: The SQL DDL query string.
        """
        self.execute_with_feedback(ddl)

    def execute_with_feedback(self, query: str, params: tuple[object, ...] | None = None) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute synchronously with feedback.

        Args:
            query: The SQL query.
            params: Parameters for the query.

        Returns:
            A tuple containing success boolean, results list, and error string if any.
        """
        try:
            if hasattr(self.conn, "cursor"):
                cursor = self.conn.cursor()
                cursor.execute(query, params or ())
                results = cursor.fetchall() if getattr(cursor, "description", None) is not None else []
                if hasattr(cursor, "close"):
                    cursor.close()
                return (True, results, None)
            else:
                results = self.conn.execute(query, params or ()).fetchall()
                return (True, results, None)
        except Exception as e:  # noqa: BLE001
            return (False, [], str(e))

    async def execute_with_feedback_async(self, query: str, params: tuple[object, ...] | None = None) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute asynchronously with feedback.

        Args:
            query: The SQL query.
            params: Parameters for the query.

        Returns:
            A tuple containing success boolean, results list, and error string if any.
        """
        import asyncio

        return await asyncio.to_thread(self.execute_with_feedback, query, params)

    def execute_query(self, query: str, params: tuple[object, ...] | None = None) -> list[tuple[JSONPrimitive, ...]]:
        """Execute synchronously.

        Args:
            query: The SQL query.
            params: Parameters for the query.

        Returns:
            The results list.
        """
        try:
            if hasattr(self.conn, "cursor"):
                cursor = self.conn.cursor()
                cursor.execute(query, params or ())
                results = cursor.fetchall() if getattr(cursor, "description", None) is not None else []
                if hasattr(cursor, "close"):
                    cursor.close()
                return results
            else:
                return self.conn.execute(query, params or ()).fetchall()
        except Exception as e:  # noqa: BLE001
            logger.debug("Query execution failed: %s", e)
            return []

    async def execute_query_async(self, query: str, params: tuple[object, ...] | None = None) -> list[tuple[JSONPrimitive, ...]]:
        """Execute asynchronously.

        Args:
            query: The SQL query.
            params: Parameters for the query.

        Returns:
            The results list.
        """
        import asyncio

        return await asyncio.to_thread(self.execute_query, query, params)

    def close(self) -> None:
        """Close connection."""
        if hasattr(self.conn, "close"):  # pragma: no cover
            self.conn.close()
