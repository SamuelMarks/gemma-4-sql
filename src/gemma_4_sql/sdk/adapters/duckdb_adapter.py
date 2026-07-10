"""DuckDB adapter."""

from __future__ import annotations

import asyncio
import logging

from gemma_4_sql.backends.lazy_loader import LazyLoader

from .base import DatabaseAdapter

logger = logging.getLogger(__name__)
duckdb = LazyLoader("duckdb").get_module()


class DuckDBAdapter(DatabaseAdapter):
    """Adapter for DuckDB."""

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Return the exception classes."""
        import duckdb

        return (duckdb.Error,)

    def connect(self) -> object:
        """Connect synchronously.

        Returns:
            The execution result.
        """
        if duckdb is None:
            msg = "duckdb is required. Install with `pip install duckdb`."
            raise ImportError(msg)
        kwargs = self.db_kwargs.copy()
        if self.read_only and self.db_path != ":memory:":
            kwargs["read_only"] = True
        return duckdb.connect(self.db_path, **kwargs)

    async def connect_async(self) -> object:
        """Connect asynchronously.

        Returns:
            object: The resulting output from the operation.

        """
        return self.conn

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL to set up schema."""
        self.conn.execute(ddl)

    async def execute_with_feedback_async(self, query: str, params: tuple[object, ...] | None = None) -> tuple[bool, list[tuple[object, ...]], str | None]:
        """Execute asynchronously with feedback.

        Returns:
            object: The resulting output from the operation.

        """
        try:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, lambda: self.conn.execute(query, params or ()).fetchall())
        except self.error_classes as e:
            return (False, [], str(e))
        else:
            return (True, results, None)

    async def execute_query_async(self, query: str, params: tuple[object, ...] | None = None) -> list[tuple[object, ...]]:
        """Execute asynchronously.

        Returns:
            object: The resulting output from the operation.

        """
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self.conn.execute(query, params or ()).fetchall())
        except self.error_classes as e:
            logger.debug("Async Query execution failed: %s", e)
            return []
