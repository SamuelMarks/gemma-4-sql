"""Live Database Evaluation Engine for Text-to-SQL execution accuracy."""

from __future__ import annotations

import asyncio
import collections
import logging
import math
import re
from typing import TYPE_CHECKING, cast

from gemma_4_sql.exceptions import ConfigurationError, SQLSafetyViolationError
from gemma_4_sql.sdk.adapters.duckdb_adapter import DuckDBAdapter
from gemma_4_sql.sdk.adapters.postgres_adapter import PostgresAdapter
from gemma_4_sql.sdk.adapters.snowflake_adapter import SnowflakeAdapter
from gemma_4_sql.sdk.adapters.sqlite_adapter import SQLiteAdapter

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONPrimitive

logger = logging.getLogger(__name__)
_ADAPTERS = {
    "sqlite": SQLiteAdapter,
    "postgresql": PostgresAdapter,
    "snowflake": SnowflakeAdapter,
    "duckdb": DuckDBAdapter,
}

_DANGEROUS_KEYWORDS = frozenset({
    "DROP",
    "CREATE",
    "ALTER",
    "TRUNCATE",
    "RENAME",
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "UPSERT",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "COPY",
    "INSTALL",
    "LOAD",
    "VACUUM",
    "EXECUTE",
    "CALL",
    "GRANT",
    "REVOKE",
})

_SQL_TOKEN_PATTERN = re.compile(
    r"""
    ('(?:''|[^'])*')            | # Single-quoted string literal
    ("(?:""|[^"])*")            | # Double-quoted identifier/literal
    (`[^`]*`)                   | # Backtick-quoted identifier
    (\[[^\]]*\])                | # Bracket-quoted identifier
    (/\*[\s\S]*?\*/)            | # Block comment
    (--[^
]*)                  # Single-line comment
    """,
    re.VERBOSE,
)


def _strip_comments_and_literals(query: str) -> str:
    """Replace SQL comments and quoted literals with spaces in a single lexical pass.

    By matching string literals and comments simultaneously from left to right,
    comment markers (`--`, `/*`) inside string literals and quotes inside comments
    are handled accurately without corrupting surrounding tokens.

    Args:
        query: Raw SQL query string.

    Returns:
        SQL string with all comments and quoted segments replaced by a single space.

    """
    return _SQL_TOKEN_PATTERN.sub(" ", query)


class LiveDatabaseEngine:
    """Execute SQL queries against an underlying database engine.

    Measures execution accuracy (EX) of generated Text-to-SQL queries across
    SQLite, PostgreSQL, Snowflake, and DuckDB while enforcing read-only safety.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        ddl: str | None = None,
        db_type: str = "sqlite",
        **kwargs: object,
    ) -> None:
        """Initialize the LiveDatabaseEngine.

        Args:
            db_path: The file path or URI to the database.
            ddl: Optional Data Definition Language (DDL) string to initialize schema.
            db_type: The database dialect ('sqlite', 'postgresql', 'snowflake', 'duckdb').
            **kwargs: Optional keyword arguments (`db_kwargs`, `read_only`).

        Raises:
            ConfigurationError: If an unsupported `db_type` is provided.

        """
        self.db_path = db_path
        self.db_type = db_type.lower()
        self.db_kwargs: dict[str, object] = cast(dict[str, object], kwargs.get("db_kwargs", {}))
        self.read_only: bool = bool(kwargs.get("read_only", True))
        adapter_cls = _ADAPTERS.get(self.db_type)
        if adapter_cls is None:
            msg = f"Unsupported db_type: {self.db_type}"
            raise ConfigurationError(msg)
        self.adapter = adapter_cls(self.db_path, self.db_kwargs, read_only=self.read_only)
        self.conn = self.adapter.conn
        if ddl:
            self.setup_schema(ddl)

    def connect(self) -> object:
        """Connect to the underlying database synchronously.

        Returns:
            The active database connection object.

        """
        return self.adapter.connect()

    async def connect_async(self) -> object:
        """Asynchronously connect to the underlying database.

        Returns:
            The active asynchronous database connection object.

        """
        return await self.adapter.connect_async()

    def _validate_safety(self, query: str) -> None:
        """Ensure the query is safe to execute when `read_only` is True.

        Args:
            query: SQL query string to inspect.

        Raises:
            SQLSafetyViolationError: If mutating statements or stacked queries are attempted in read-only mode.

        """
        if not self.read_only:
            return

        cleaned = _strip_comments_and_literals(query)
        statements = [stmt.strip() for stmt in cleaned.split(";") if stmt.strip()]
        if len(statements) > 1:
            msg = "Safety Violation: Multiple stacked SQL statements are not allowed in read-only mode."
            raise SQLSafetyViolationError(msg)

        words = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", cleaned.upper())
        for word in words:
            if word in _DANGEROUS_KEYWORDS:
                msg = f"Safety Violation: Mutating statements ({word}) are not allowed in read-only mode."
                raise SQLSafetyViolationError(msg)

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

    def execute_with_feedback(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute a query and return execution success status, results, and error message.

        Args:
            query: SQL query string to execute.
            params: Optional tuple of query parameters.

        Returns:
            Tuple of `(success, fetched_rows, optional_error_message)`.

        """
        try:
            self._validate_safety(query)
            return self.adapter.execute_with_feedback(query, params)
        except PermissionError as e:
            return (False, [], str(e))

    async def execute_with_feedback_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Asynchronously execute a query and return execution success status, results, and error message.

        Args:
            query: SQL query string to execute.
            params: Optional tuple of query parameters.

        Returns:
            Tuple of `(success, fetched_rows, optional_error_message)`.

        """
        try:
            self._validate_safety(query)
            return await self.adapter.execute_with_feedback_async(query, params)
        except PermissionError as e:
            return (False, [], str(e))

    def execute_query(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a query synchronously and return the fetched results.

        Args:
            query: SQL query string to execute.
            params: Optional tuple of query parameters.

        Returns:
            List of fetched result tuples.

        Raises:
            SQLSafetyViolationError: If query violates read-only mode.
            DatabaseExecutionError: If query execution fails.

        """
        self._validate_safety(query)
        return self.adapter.execute_query(query, params)

    async def execute_query_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Asynchronously execute a query and return the fetched results.

        Args:
            query: SQL query string to execute.
            params: Optional tuple of query parameters.

        Returns:
            List of fetched result tuples.

        Raises:
            SQLSafetyViolationError: If query violates read-only mode.
            DatabaseExecutionError: If query execution fails.

        """
        self._validate_safety(query)
        return await self.adapter.execute_query_async(query, params)

    def _has_top_level_order_by(self, query: str) -> bool:
        """Detect presence of a top-level `ORDER BY` clause outside parentheses.

        Args:
            query: SQL query string.

        Returns:
            True if a top-level `ORDER BY` clause is present.

        """
        cleaned = _strip_comments_and_literals(query)
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
        """Normalize row values to handle float tolerance and `NULL`/`NaN` equivalents.

        Args:
            row: A tuple of raw database column values.

        Returns:
            A normalized tuple suitable for equality and multiset comparison.

        """
        normalized: list[JSONPrimitive] = []
        for val in row:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                normalized.append(None)
            elif isinstance(val, float):
                if val == 0.0:
                    normalized.append(0.0)
                else:
                    normalized.append(round(val, -math.floor(math.log10(abs(val))) + 4))
            else:
                normalized.append(val)
        return tuple(normalized)

    def _compare_results(
        self,
        pred_results: list[tuple[JSONPrimitive, ...]],
        truth_results: list[tuple[JSONPrimitive, ...]],
        strict_order: bool,
    ) -> bool:
        """Compare two sets of database result rows.

        Args:
            pred_results: Result rows from the predicted SQL query.
            truth_results: Result rows from the ground-truth SQL query.
            strict_order: Whether row ordering must match exactly.

        Returns:
            True if the result sets match under the specified ordering rule.

        """
        norm_pred = [self._normalize_row(r) for r in pred_results]
        norm_truth = [self._normalize_row(r) for r in truth_results]

        if strict_order:
            return norm_pred == norm_truth

        return collections.Counter(norm_pred) == collections.Counter(norm_truth)

    def compare_queries(self, predicted_sql: str, ground_truth_sql: str) -> bool:
        """Compare the execution results of two SQL queries synchronously.

        If either the predicted query or the ground-truth query fails to execute,
        returns `False`.

        Args:
            predicted_sql: The generated SQL query.
            ground_truth_sql: The reference SQL query.

        Returns:
            True if both queries execute validly and produce equivalent results.

        """
        pred_ok, pred_results, _ = self.execute_with_feedback(predicted_sql)
        truth_ok, truth_results, _ = self.execute_with_feedback(ground_truth_sql)
        if not pred_ok or not truth_ok:
            return False
        strict_order = self._has_top_level_order_by(ground_truth_sql)
        return self._compare_results(pred_results, truth_results, strict_order)

    async def compare_queries_async(self, predicted_sql: str, ground_truth_sql: str) -> bool:
        """Asynchronously compare the execution results of two SQL queries.

        If either the predicted query or the ground-truth query fails to execute,
        returns `False`.

        Args:
            predicted_sql: The generated SQL query.
            ground_truth_sql: The reference SQL query.

        Returns:
            True if both queries execute validly and produce equivalent results.

        """
        (pred_ok, pred_results, _), (truth_ok, truth_results, _) = await asyncio.gather(
            self.execute_with_feedback_async(predicted_sql),
            self.execute_with_feedback_async(ground_truth_sql),
        )
        if not pred_ok or not truth_ok:
            return False
        strict_order = self._has_top_level_order_by(ground_truth_sql)
        return self._compare_results(pred_results, truth_results, strict_order)

    def close(self) -> None:
        """Close the underlying database connection."""
        self.adapter.close()


__all__ = ["LiveDatabaseEngine"]
