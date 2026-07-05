# Copyright 2024
"""Base adapter protocol for database connections."""

from __future__ import annotations

import logging
import typing

if typing.TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONPrimitive

logger = logging.getLogger(__name__)


class DatabaseAdapter:
    """Protocol for database dialect adapters."""

    def __init__(self, db_path: str, db_kwargs: dict[str, object], *, read_only: bool = False) -> None:
        """Initialize adapter."""
        self.db_path = db_path
        self.db_kwargs = db_kwargs
        self.read_only = read_only
        self.conn = self.connect()

    def connect(self) -> object:
        """Connect synchronously."""
        raise NotImplementedError  # pragma: no cover

    async def connect_async(self) -> object:
        """Connect asynchronously."""
        raise NotImplementedError  # pragma: no cover

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL to set up schema."""
        raise NotImplementedError  # pragma: no cover

    def execute_with_feedback(self, query: str, params: tuple[object, ...] | None = None) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute synchronously with feedback."""
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
        """Execute asynchronously with feedback."""
        raise NotImplementedError  # pragma: no cover

    def execute_query(self, query: str, params: tuple[object, ...] | None = None) -> list[tuple[JSONPrimitive, ...]]:
        """Execute synchronously."""
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
        """Execute asynchronously."""
        raise NotImplementedError  # pragma: no cover

    def close(self) -> None:
        """Close connection."""
        if hasattr(self.conn, "close"):  # pragma: no cover
            self.conn.close()
