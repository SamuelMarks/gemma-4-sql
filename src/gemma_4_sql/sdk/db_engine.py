"""Live Database Evaluation Engine for Text-to-SQL execution accuracy."""

from __future__ import annotations

import asyncio
import collections
import collections.abc
import logging
import math
import re
from typing import TYPE_CHECKING, Any

from gemma_4_sql.exceptions import ConfigurationError
from gemma_4_sql.sdk.adapters.base import (
    DatabaseAdapter,
    validate_readonly_sql,
)
from gemma_4_sql.sdk.adapters.base import (
    strip_comments_and_literals as _strip_comments_and_literals,
)
from gemma_4_sql.sdk.adapters.duckdb_adapter import DuckDBAdapter
from gemma_4_sql.sdk.adapters.postgres_adapter import PostgresAdapter
from gemma_4_sql.sdk.adapters.snowflake_adapter import SnowflakeAdapter
from gemma_4_sql.sdk.adapters.sqlite_adapter import SQLiteAdapter

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    from gemma_4_sql.type_hints import JSONPrimitive

logger = logging.getLogger(__name__)
_ADAPTERS: dict[str, type[DatabaseAdapter]] = {
    "sqlite": SQLiteAdapter,
    "postgresql": PostgresAdapter,
    "snowflake": SnowflakeAdapter,
    "duckdb": DuckDBAdapter,
}


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
        raw_db_kwargs = kwargs.get("db_kwargs")
        self.db_kwargs: dict[str, object] = dict(raw_db_kwargs) if isinstance(raw_db_kwargs, collections.abc.Mapping) else {}
        self.read_only: bool = bool(kwargs.get("read_only", True))
        adapter_cls = _ADAPTERS.get(self.db_type)
        if adapter_cls is None:
            msg = f"Unsupported db_type: {self.db_type}"
            raise ConfigurationError(msg)
        self.adapter: DatabaseAdapter = adapter_cls(self.db_path, self.db_kwargs, read_only=self.read_only)
        self.conn = self.adapter.conn
        self._pending_ddl: str | None = ddl if kwargs.get("defer_ddl") else None
        if ddl and not kwargs.get("defer_ddl"):
            self.setup_schema(ddl)

    def connect(self) -> object:
        """Connect to the underlying database synchronously.

        Returns:
            The active database connection object.

        """
        return self.adapter.connect()

    async def connect_async(self) -> object:
        """Asynchronously connect to the underlying database and apply any deferred DDL.

        Returns:
            The active asynchronous database connection object.

        """
        if self._pending_ddl:
            ddl_to_run = self._pending_ddl
            self._pending_ddl = None
            await self.setup_schema_async(ddl_to_run)
        return await self.adapter.connect_async()

    def _validate_safety(self, query: str) -> None:
        """Ensure the query is safe to execute when `read_only` is True.

        Args:
            query: SQL query string to inspect.

        Raises:
            SQLSafetyViolationError: If mutating statements or stacked queries are attempted in read-only mode.

        """
        validate_readonly_sql(query, read_only=self.read_only)

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL statements to construct the database schema.

        Args:
            ddl: The Data Definition Language (DDL) string.

        """
        self.adapter.setup_schema(ddl)

    async def setup_schema_async(self, ddl: str) -> None:
        """Asynchronously execute DDL statements to construct the database schema without blocking the event loop.

        Args:
            ddl: The Data Definition Language (DDL) string.

        """
        await asyncio.to_thread(self.setup_schema, ddl)

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

    def has_top_level_order_by(self, query: str) -> bool:
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

    _has_top_level_order_by = has_top_level_order_by

    def normalize_row(self, row: tuple[JSONPrimitive, ...]) -> tuple[JSONPrimitive, ...]:
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

    _normalize_row = normalize_row

    def compare_results(
        self,
        pred_results: Sequence[tuple[JSONPrimitive, ...]],
        truth_results: Sequence[tuple[JSONPrimitive, ...]],
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

        def _tag_row(row: tuple[JSONPrimitive, ...]) -> tuple[tuple[str, JSONPrimitive], ...]:
            """Tag each normalized cell with its canonical type category to prevent cross-type collisions."""
            norm = self.normalize_row(row)
            tagged: list[tuple[str, JSONPrimitive]] = []
            for cell in norm:
                if cell is None:
                    tagged.append(("null", None))
                elif isinstance(cell, bool):
                    tagged.append(("bool", cell))
                elif isinstance(cell, (int, float)):
                    tagged.append(("num", float(cell)))
                else:
                    tagged.append(("str", str(cell)))
            return tuple(tagged)

        norm_pred = [_tag_row(r) for r in pred_results]
        norm_truth = [_tag_row(r) for r in truth_results]

        if strict_order:
            return norm_pred == norm_truth

        return collections.Counter(norm_pred) == collections.Counter(norm_truth)

    _compare_results = compare_results

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
        strict_order = self.has_top_level_order_by(ground_truth_sql)
        return self.compare_results(pred_results, truth_results, strict_order)

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
        strict_order = self.has_top_level_order_by(ground_truth_sql)
        return self.compare_results(pred_results, truth_results, strict_order)

    async def close_async(self) -> None:
        """Asynchronously close the underlying database connection."""
        close_async_fn = getattr(self.adapter, "close_async", None)
        if callable(close_async_fn):
            res: Any = close_async_fn()
            await res
        self.adapter.close()

    def close(self) -> None:
        """Close the underlying database connection."""
        self.adapter.close()

    def __enter__(self) -> Self:
        """Enter synchronous context manager."""
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        """Exit synchronous context manager and close database connection."""
        self.close()

    async def __aenter__(self) -> Self:
        """Enter asynchronous context manager."""
        return self

    async def __aexit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        """Exit asynchronous context manager and close database connection."""
        await self.close_async()


__all__ = ["LiveDatabaseEngine"]
