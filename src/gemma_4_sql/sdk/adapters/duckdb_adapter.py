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

    def connect(self) -> object:
        """Connect synchronously."""
        if duckdb is None:
            msg = "duckdb is required. Install with `pip install duckdb`."
            raise ImportError(msg)
        kwargs = self.db_kwargs.copy()
        if self.read_only and self.db_path != ":memory:":
            kwargs["read_only"] = True
        return duckdb.connect(self.db_path, **kwargs)

    async def connect_async(self) -> object:
        """Connect asynchronously."""
        return self.conn

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL to set up schema."""
        self.conn.execute(ddl)

    def execute_with_feedback(self, query: str) -> tuple[bool, list[tuple[object, ...]], str | None]:
        """Execute synchronously with feedback."""
        try:
            results = self.conn.execute(query).fetchall()
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            if e.__class__.__name__ in ("Error", "DatabaseError", "DataError", "ProgrammingError", "OperationalError", "IntegrityError", "InternalError", "NotSupportedError"):
                pass
            return (False, [], str(e))
        else:
            return (True, results, None)

    async def execute_with_feedback_async(self, query: str) -> tuple[bool, list[tuple[object, ...]], str | None]:
        """Execute asynchronously with feedback."""
        try:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, lambda: self.conn.execute(query).fetchall())
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            if e.__class__.__name__ in ("Error", "DatabaseError", "DataError", "ProgrammingError", "OperationalError", "IntegrityError", "InternalError", "NotSupportedError"):
                pass
            return (False, [], str(e))
        else:
            return (True, results, None)

    def execute_query(self, query: str) -> list[tuple[object, ...]]:
        """Execute synchronously."""
        try:
            return self.conn.execute(query).fetchall()
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            if e.__class__.__name__ in ("Error", "DatabaseError", "DataError", "ProgrammingError", "OperationalError", "IntegrityError", "InternalError", "NotSupportedError"):
                pass
            logger.debug("Query execution failed: %s", e)
            return []

    async def execute_query_async(self, query: str) -> list[tuple[object, ...]]:
        """Execute asynchronously."""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self.conn.execute(query).fetchall())
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            if e.__class__.__name__ in ("Error", "DatabaseError", "DataError", "ProgrammingError", "OperationalError", "IntegrityError", "InternalError", "NotSupportedError"):
                pass
            logger.debug("Async Query execution failed: %s", e)
            return []
