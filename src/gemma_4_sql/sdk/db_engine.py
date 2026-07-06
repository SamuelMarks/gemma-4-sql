"""Live Database Evaluation Engine for Text-to-SQL execution accuracy."""

from __future__ import annotations

import asyncio
import logging
import re
import typing

from gemma_4_sql.sdk.adapters.duckdb_adapter import DuckDBAdapter
from gemma_4_sql.sdk.adapters.postgres_adapter import PostgresAdapter
from gemma_4_sql.sdk.adapters.snowflake_adapter import SnowflakeAdapter
from gemma_4_sql.sdk.adapters.sqlite_adapter import SQLiteAdapter

if typing.TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONPrimitive
logger = logging.getLogger(__name__)
_ADAPTERS = {"sqlite": SQLiteAdapter, "postgresql": PostgresAdapter, "snowflake": SnowflakeAdapter, "duckdb": DuckDBAdapter}


class LiveDatabaseEngine:
    """Execute SQL queries against an underlying database engine.

    to measure execution accuracy (EX) of generated Text-to-SQL queries.
    Supports SQLite, PostgreSQL, Snowflake, and DuckDB.
    """

    def __init__(self, db_path: str = ":memory:", ddl: str | None = None, db_type: str = "sqlite", **kwargs: object) -> None:
        """Initialize the LiveDatabaseEngine.

        Args:
            db_path: The file path to the database.
            ddl: The Data Definition Language (DDL) string.
            db_type: The string representing the db type.
            **kwargs: Additional keyword arguments.
        """
        self.db_path = db_path
        self.db_type = db_type.lower()
        self.db_kwargs = kwargs.get("db_kwargs") or {}
        self.read_only = kwargs.get("read_only", True)
        adapter_cls = _ADAPTERS.get(self.db_type)
        if adapter_cls is None:
            msg = f"Unsupported db_type: {self.db_type}"
            raise ValueError(msg)
        self.adapter = adapter_cls(self.db_path, self.db_kwargs, read_only=self.read_only)
        self.conn = self.adapter.conn
        if ddl:
            old_ro = self.read_only
            self.read_only = False
            self.adapter.read_only = False
            try:
                self.setup_schema(ddl)
            finally:
                self.read_only = old_ro
                self.adapter.read_only = old_ro

    def connect(self) -> object:
        """Connect to database.

        Returns:
            object: The resulting output from the operation.

        """
        return self.adapter.connect()

    async def connect_async(self) -> object:
        """Asynchronously connect to database.

        Returns:
            object: The resulting output from the operation.

        """
        return await self.adapter.connect_async()

    def _validate_safety(self, query: str) -> None:
        """Ensure the query is safe to execute if read_only is True.

        Raises:
        PermissionError: If the operation encounters an unexpected PermissionError.

        """
        if not self.read_only:
            return
        dangerous_patterns = ["\\bDROP\\b", "\\bDELETE\\b", "\\bUPDATE\\b", "\\bINSERT\\b", "\\bALTER\\b", "\\bTRUNCATE\\b", "\\bGRANT\\b", "\\bREVOKE\\b"]
        upper_query = query.upper()
        for pattern in dangerous_patterns:
            if re.search(pattern, upper_query):
                msg = f"Safety Violation: Mutating statements ({pattern}) are not allowed in read-only mode."
                raise PermissionError(msg)

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL statements to construct the database schema."""
        self._validate_safety(ddl)
        self.adapter.setup_schema(ddl)

    def execute_with_feedback(self, query: str, params: tuple[object, ...] | None = None) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute a query and returns execution success status, results, and error message.

        Args:
            ddl: The Data Definition Language (DDL) string.
        """
        try:
            self._validate_safety(query)
            return self.adapter.execute_with_feedback(query, params)
        except PermissionError as e:
            return (False, [], str(e))

    async def execute_with_feedback_async(self, query: str, params: tuple[object, ...] | None = None) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Asynchronously execute a query and returns execution success status, results, and error message.

        Returns:
            object: The resulting output from the operation.

        """
        try:
            self._validate_safety(query)
            return await self.adapter.execute_with_feedback_async(query, params)
        except PermissionError as e:
            return (False, [], str(e))

    def execute_query(self, query: str, params: tuple[object, ...] | None = None) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a query and returns the fetched results.

        Returns:
            object: The resulting output from the operation.

        """
        self._validate_safety(query)
        return self.adapter.execute_query(query, params)

    async def execute_query_async(self, query: str, params: tuple[object, ...] | None = None) -> list[tuple[JSONPrimitive, ...]]:
        """Asynchronously execute a query and returns the fetched results.

        Returns:
            object: The resulting output from the operation.

        """
        self._validate_safety(query)
        return await self.adapter.execute_query_async(query, params)

    def compare_queries(self, predicted_sql: str, ground_truth_sql: str) -> bool:
        """Compare the execution results of two queries.

        Returns:
            object: The resulting output from the operation.

        """
        pred_results = self.execute_query(predicted_sql)
        truth_results = self.execute_query(ground_truth_sql)
        return pred_results == truth_results

    async def compare_queries_async(self, predicted_sql: str, ground_truth_sql: str) -> bool:
        """Asynchronously compare the execution results of two queries.

        Returns:
            object: The resulting output from the operation.

        """
        (pred_results, truth_results) = await asyncio.gather(self.execute_query_async(predicted_sql), self.execute_query_async(ground_truth_sql))
        return pred_results == truth_results

    def close(self) -> None:
        """Close the database connection."""
        self.adapter.close()
