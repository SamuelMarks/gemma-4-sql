"""Base adapter protocol for database connections."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONPrimitive


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
        raise NotImplementedError

    async def connect_async(self) -> object:
        """Connect asynchronously."""
        raise NotImplementedError

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL to set up schema."""
        raise NotImplementedError

    def execute_with_feedback(self, query: str) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute synchronously with feedback."""
        raise NotImplementedError

    async def execute_with_feedback_async(self, query: str) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute asynchronously with feedback."""
        raise NotImplementedError

    def execute_query(self, query: str) -> list[tuple[JSONPrimitive, ...]]:
        """Execute synchronously."""
        raise NotImplementedError

    async def execute_query_async(self, query: str) -> list[tuple[JSONPrimitive, ...]]:
        """Execute asynchronously."""
        raise NotImplementedError

    def close(self) -> None:
        """Close connection."""
        if hasattr(self.conn, "close"):
            self.conn.close()
