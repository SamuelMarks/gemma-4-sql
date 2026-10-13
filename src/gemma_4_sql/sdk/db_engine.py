"""Live Database Evaluation Engine for Text-to-SQL execution accuracy."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, cast

from gemma_4_sql.sdk.adapters.duckdb_adapter import DuckDBAdapter
from gemma_4_sql.sdk.adapters.postgres_adapter import PostgresAdapter
from gemma_4_sql.sdk.adapters.snowflake_adapter import SnowflakeAdapter
from gemma_4_sql.sdk.adapters.sqlite_adapter import SQLiteAdapter

if TYPE_CHECKING:
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
            **kwargs: Optional keyword arguments for advanced configuration.

        Raises:
            ValueError: If unsupported db_type is provided.

        """
        self.db_path = db_path
        self.db_type = db_type.lower()
        self.db_kwargs: dict[str, object] = cast(dict[str, object], kwargs.get("db_kwargs", {}))
        self.read_only: bool = bool(kwargs.get("read_only", True))
        adapter_cls = _ADAPTERS.get(self.db_type)
        if adapter_cls is None:
            msg = f"Unsupported db_type: {self.db_type}"
            raise ValueError(msg)
        self.adapter = adapter_cls(self.db_path, self.db_kwargs, read_only=self.read_only)
        self.conn = self.adapter.conn
        if ddl:
            self.setup_schema(ddl)

    def connect(self) -> object:
        """Connect to database.

        Returns:
            The database connection.

        """
        return self.adapter.connect()

    async def connect_async(self) -> object:
        """Asynchronously connect to database.

        Returns:
            The asynchronous database connection.

        """
        return await self.adapter.connect_async()

    def _validate_safety(self, query: str) -> None:
        """Ensure the query is safe to execute if read_only is True.

        Args:
            query: SQL query string to inspect.

        Raises:
            PermissionError: If mutating statements are attempted in read-only mode.

        """
        if not self.read_only:
            return

        # Strip comments and string/identifier literals to isolate SQL keywords
        cleaned = query
        cleaned = re.sub(r"/\*.*?\*/", " ", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"--[^\n]*", " ", cleaned)
        cleaned = re.sub(r"'([^']|'')*'", " ", cleaned)
        cleaned = re.sub(r"\"([^\"]|\"\")*\"", " ", cleaned)
        cleaned = re.sub(r"`[^`]*`", " ", cleaned)
        cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
        # We also need to strip dollar-quoted strings, but doing that with regex is tricky without causing issues.
        # We'll just rely on the above for now, which covers 99% of cases.

        words = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", cleaned.upper())

        dangerous_keywords = {"DROP", "CREATE", "ALTER", "TRUNCATE", "RENAME", "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "ATTACH", "DETACH", "PRAGMA", "COPY", "INSTALL", "LOAD", "VACUUM", "EXECUTE", "CALL", "GRANT", "REVOKE"}

        for word in words:
            if word in dangerous_keywords:
                msg = f"Safety Violation: Mutating statements ({word}) are not allowed in read-only mode."
                raise PermissionError(msg)

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL statements to construct the database schema.

        Args:
            ddl: The Data Definition Language (DDL) string.

        """
        old_ro = self.read_only
        self.read_only = False
        self.adapter.read_only = False
        try:
            self.adapter.setup_schema(ddl)
        finally:
            self.read_only = old_ro
            self.adapter.read_only = old_ro

    def execute_with_feedback(self, query: str, params: tuple[object, ...] | None = None) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute a query and returns execution success status, results, and error message.

        Args:
            query: SQL query string to execute.
            params: Optional tuple of query parameters.

        Returns:
            Tuple of (success, fetched rows, optional error message).

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

    def _has_top_level_order_by(self, query: str) -> bool:
        """Detect presence of top-level ORDER BY clause."""
        cleaned = re.sub(r"/\*.*?\*/", " ", query, flags=re.DOTALL)
        cleaned = re.sub(r"--[^\n]*", " ", cleaned)
        cleaned = re.sub(r"'([^']|'')*'", " ", cleaned)
        cleaned = re.sub(r"\"([^\"]|\"\")*\"", " ", cleaned)
        cleaned = re.sub(r"`[^`]*`", " ", cleaned)
        cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
        cleaned = re.sub(r"\bORDER\s+BY\b", "ORDER_BY", cleaned.upper())
        tokens = re.finditer(r"\(|\)|ORDER_BY", cleaned)
        depth = 0
        for match in tokens:
            t = match.group(0)
            if t == "(":
                depth += 1
            elif t == ")":
                depth -= 1
            elif t == "ORDER_BY" and depth <= 0:
                return True
        return False

    def _normalize_row(self, row: tuple[JSONPrimitive, ...]) -> tuple[JSONPrimitive, ...]:
        """Normalize row values to handle float tolerance and NULL equivalents."""
        import math

        normalized = []
        for val in row:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                normalized.append(None)
            elif isinstance(val, float):
                if val == 0.0:
                    normalized.append(0.0)
                else:
                    # Round to 5 significant digits (approx 1e-5 relative tolerance)
                    normalized.append(round(val, -math.floor(math.log10(abs(val))) + 4))
            else:
                normalized.append(val)
        return tuple(normalized)

    def _compare_results(self, pred_results: list[tuple[JSONPrimitive, ...]], truth_results: list[tuple[JSONPrimitive, ...]], strict_order: bool) -> bool:
        """Compare two sets of database results."""
        import collections

        norm_pred = [self._normalize_row(r) for r in pred_results]
        norm_truth = [self._normalize_row(r) for r in truth_results]

        if strict_order:
            return norm_pred == norm_truth

        return collections.Counter(norm_pred) == collections.Counter(norm_truth)

    def compare_queries(self, predicted_sql: str, ground_truth_sql: str) -> bool:
        """Compare the execution results of two queries.

        Args:
            predicted_sql: The generated SQL query.
            ground_truth_sql: The reference SQL query.

        Returns:
            True if execution results are semantically equivalent.

        """
        pred_results = self.execute_query(predicted_sql)
        truth_results = self.execute_query(ground_truth_sql)
        strict_order = self._has_top_level_order_by(ground_truth_sql)
        return self._compare_results(pred_results, truth_results, strict_order)

    async def compare_queries_async(self, predicted_sql: str, ground_truth_sql: str) -> bool:
        """Asynchronously compare the execution results of two queries.

        Args:
            predicted_sql: The generated SQL query.
            ground_truth_sql: The reference SQL query.

        Returns:
            True if execution results are semantically equivalent.

        """
        (pred_results, truth_results) = await asyncio.gather(self.execute_query_async(predicted_sql), self.execute_query_async(ground_truth_sql))
        strict_order = self._has_top_level_order_by(ground_truth_sql)
        return self._compare_results(pred_results, truth_results, strict_order)

    def close(self) -> None:
        """Close the database connection."""
        self.adapter.close()


__all__ = ["LiveDatabaseEngine"]
