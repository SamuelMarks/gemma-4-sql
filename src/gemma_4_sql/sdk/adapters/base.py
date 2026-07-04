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
        raise NotImplementedError  # pragma: no cover

    async def execute_with_feedback_async(self, query: str, params: tuple[object, ...] | None = None) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute asynchronously with feedback."""
        raise NotImplementedError  # pragma: no cover

    def execute_query(self, query: str, params: tuple[object, ...] | None = None) -> list[tuple[JSONPrimitive, ...]]:
        """Execute synchronously."""
        raise NotImplementedError  # pragma: no cover

    async def execute_query_async(self, query: str, params: tuple[object, ...] | None = None) -> list[tuple[JSONPrimitive, ...]]:
        """Execute asynchronously."""
        raise NotImplementedError  # pragma: no cover

    def close(self) -> None:
        """Close connection."""
        if hasattr(self.conn, "close"):  # pragma: no cover
            self.conn.close()
