"""DuckDB adapter."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONPrimitive

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

        Raises:
            ImportError: If duckdb is not installed.
        """
        if "existing_conn" in self.db_kwargs and self.db_kwargs["existing_conn"] is not None:
            return self.db_kwargs["existing_conn"]
        if "conn" in self.db_kwargs and self.db_kwargs["conn"] is not None:
            return self.db_kwargs["conn"]
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
        """Execute DDL to set up schema.

        Args:
            ddl: Schema DDL string to execute.
        """
        if "existing_conn" in self.db_kwargs or "conn" in self.db_kwargs:
            return
        cast(Any, self.conn).execute(ddl)

    def close(self) -> None:
        """Close connection if not an externally provided connection."""
        if "existing_conn" in self.db_kwargs or "conn" in self.db_kwargs:
            return
        super().close()

    async def execute_with_feedback_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute asynchronously with feedback.

        Returns:
            object: The resulting output from the operation.

        """
        try:
            loop = asyncio.get_running_loop()

            def _exec() -> list[tuple[JSONPrimitive, ...]]:
                """Execute query on thread-safe cursor.

                Returns:
                    Query result rows.
                """
                conn_obj = cast(Any, self.conn)
                cur = conn_obj.cursor() if callable(getattr(conn_obj, "cursor", None)) else conn_obj
                return cast("list[tuple[JSONPrimitive, ...]]", cur.execute(query, params or ()).fetchall())

            results = await loop.run_in_executor(None, _exec)
        except self.error_classes as e:
            return (False, [], str(e))
        else:
            return (True, results, None)

    async def execute_query_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute asynchronously.

        Returns:
            object: The resulting output from the operation.

        """
        try:
            loop = asyncio.get_running_loop()

            def _exec() -> list[tuple[JSONPrimitive, ...]]:
                """Execute query on thread-safe cursor.

                Returns:
                    Query result rows.
                """
                conn_obj = cast(Any, self.conn)
                cur = conn_obj.cursor() if callable(getattr(conn_obj, "cursor", None)) else conn_obj
                return cast("list[tuple[JSONPrimitive, ...]]", cur.execute(query, params or ()).fetchall())

            return await loop.run_in_executor(None, _exec)
        except self.error_classes as e:
            logger.debug("Async Query execution failed: %s", e)
            return []
