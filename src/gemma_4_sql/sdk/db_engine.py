"""Live Database Evaluation Engine for Text-to-SQL execution accuracy."""

from __future__ import annotations

import asyncio
import collections
import collections.abc
import decimal
import logging
import math
import sqlite3
import types
from collections.abc import Mapping, Sequence
from typing import Self

from gemma_4_sql.exceptions import ConfigurationError, SQLSafetyViolationError
from gemma_4_sql.sdk.adapters.base import AsyncAdapterConnection, DatabaseAdapter, DBKwargValue, SyncAdapterConnection
from gemma_4_sql.sdk.adapters.duckdb_adapter import DuckDBAdapter
from gemma_4_sql.sdk.adapters.postgres_adapter import PostgresAdapter
from gemma_4_sql.sdk.adapters.snowflake_adapter import SnowflakeAdapter
from gemma_4_sql.sdk.adapters.sqlite_adapter import SQLiteAdapter
from gemma_4_sql.sdk.sql_lexer import (
    has_top_level_order_by as _has_top_level_order_by_lexer,
)
from gemma_4_sql.sdk.sql_lexer import (
    validate_readonly_sql,
)
from gemma_4_sql.type_hints import (
    AsyncDatabaseConnectionProtocol,
    DatabaseConnectionProtocol,
    JSONPrimitive,
    JSONValue,
)

logger = logging.getLogger(__name__)
type _DialectAdapterType = type[
    SQLiteAdapter
    | PostgresAdapter
    | SnowflakeAdapter
    | DuckDBAdapter
    | DatabaseAdapter[SyncAdapterConnection, AsyncAdapterConnection]
]

_ADAPTERS: dict[str, _DialectAdapterType] = {
    "sqlite": SQLiteAdapter,
    "postgresql": PostgresAdapter,
    "snowflake": SnowflakeAdapter,
    "duckdb": DuckDBAdapter,
}

DEFAULT_FLOAT_TOLERANCE = 1e-4
DEFAULT_MAX_COMPARE_ROWS = 100_000


def _cells_equivalent(a: tuple[str, JSONPrimitive], b: tuple[str, JSONPrimitive], float_tol: float) -> bool:
    """Compare two type-tagged normalized SQL cells using relative/absolute tolerance for numbers.

    Args:
        a: First `(type_tag, value)` cell tuple.
        b: Second `(type_tag, value)` cell tuple.
        float_tol: Relative and absolute tolerance for numeric comparisons.

    Returns:
        `True` if both cells share the same type tag and equivalent value.

    """
    if a[0] != b[0]:
        return False
    if a[0] == "num" and isinstance(a[1], (int, float)) and isinstance(b[1], (int, float)):
        fa = float(a[1])
        fb = float(b[1])
        if fa == fb:
            return True
        return math.isclose(fa, fb, rel_tol=float_tol, abs_tol=float_tol)
    return a[1] == b[1]


def _rows_equivalent(
    row_a: tuple[tuple[str, JSONPrimitive], ...],
    row_b: tuple[tuple[str, JSONPrimitive], ...],
    float_tol: float,
) -> bool:
    """Compare two type-tagged result rows element-by-element under `float_tol`.

    Args:
        row_a: First tagged row tuple.
        row_b: Second tagged row tuple.
        float_tol: Relative and absolute numeric tolerance.

    Returns:
        `True` if the rows have equal arity and every column pair is equivalent.

    """
    if len(row_a) != len(row_b):
        return False
    return all(_cells_equivalent(ca, cb, float_tol) for ca, cb in zip(row_a, row_b, strict=True))


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
        **kwargs: JSONValue | Mapping[str, DBKwargValue] | DatabaseConnectionProtocol | AsyncDatabaseConnectionProtocol,
    ) -> None:
        """Initialize the LiveDatabaseEngine.

        Args:
            db_path: The file path or URI to the database.
            ddl: Optional Data Definition Language (DDL) string to initialize schema.
            db_type: The database dialect (`'sqlite'`, `'postgresql'`, `'snowflake'`, `'duckdb'`).
            **kwargs: Optional keyword arguments (`db_kwargs`, `read_only`, `defer_ddl`,
                `lazy_connect`, `float_tol`, `max_rows`).

        Raises:
            ConfigurationError: If an unsupported `db_type` is provided.

        """
        self.db_path = db_path
        self.db_type = db_type.lower()
        raw_db_kwargs = kwargs.get("db_kwargs")
        self.db_kwargs: dict[str, DBKwargValue] = (
            dict(raw_db_kwargs) if isinstance(raw_db_kwargs, collections.abc.Mapping) else {}
        )
        self.read_only: bool = bool(kwargs.get("read_only", True))
        raw_tol = kwargs.get("float_tol")
        self.float_tol: float = (
            float(raw_tol)
            if isinstance(raw_tol, (int, float)) and not isinstance(raw_tol, bool)
            else DEFAULT_FLOAT_TOLERANCE
        )
        raw_max_rows = kwargs.get("max_rows", DEFAULT_MAX_COMPARE_ROWS)
        self.max_rows: int | None = (
            int(raw_max_rows)
            if isinstance(raw_max_rows, int) and not isinstance(raw_max_rows, bool) and raw_max_rows > 0
            else None
        )

        adapter_cls = _ADAPTERS.get(self.db_type)
        if adapter_cls is None:
            msg = f"Unsupported db_type: {self.db_type}"
            raise ConfigurationError(msg)
        defer_ddl = bool(kwargs.get("defer_ddl", False))
        lazy_connect = bool(kwargs.get("lazy_connect", False)) or defer_ddl
        self.adapter = adapter_cls(self.db_path, self.db_kwargs, read_only=self.read_only, lazy_connect=lazy_connect)
        self._pending_ddl: str | None = ddl if defer_ddl else None
        if ddl and not defer_ddl:
            self.setup_schema(ddl)

    @property
    def conn(self) -> SyncAdapterConnection | sqlite3.Connection:
        """Provide the underlying synchronous connection, connecting lazily if deferred.

        Returns:
            Active synchronous database connection.

        """
        return self.adapter.conn

    def reset_connection(self) -> None:
        """Clear any cached synchronous connection without closing externally owned handles."""
        self.adapter.reset_connection()

    def connect(self) -> SyncAdapterConnection | sqlite3.Connection:
        """Connect to the underlying database synchronously.

        Returns:
            The active database connection object.

        """
        return self.adapter.connect()

    async def connect_async(self) -> AsyncAdapterConnection:
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

    def _apply_row_cap(self, rows: list[tuple[JSONPrimitive, ...]]) -> list[tuple[JSONPrimitive, ...]]:
        """Enforce `self.max_rows` upper bound on fetched result sets to prevent unbounded memory growth.

        Args:
            rows: Fetched result rows.

        Returns:
            Capped list of result rows.

        """
        if self.max_rows is not None and len(rows) > self.max_rows:
            return rows[: self.max_rows]
        return rows

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
        params: Sequence[JSONPrimitive] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute a query and return execution success status, results, and error message.

        Args:
            query: SQL query string to execute.
            params: Optional sequence of query parameters.

        Returns:
            Tuple of `(success, fetched_rows, optional_error_message)`.

        """
        try:
            self._validate_safety(query)
            ok, rows, err = self.adapter.execute_with_feedback(query, params)
            return (ok, self._apply_row_cap(rows), err)
        except SQLSafetyViolationError as e:
            return (False, [], str(e))

    async def execute_with_feedback_async(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Asynchronously execute a query and return execution success status, results, and error message.

        Args:
            query: SQL query string to execute.
            params: Optional sequence of query parameters.

        Returns:
            Tuple of `(success, fetched_rows, optional_error_message)`.

        """
        try:
            self._validate_safety(query)
            ok, rows, err = await self.adapter.execute_with_feedback_async(query, params)
            return (ok, self._apply_row_cap(rows), err)
        except SQLSafetyViolationError as e:
            return (False, [], str(e))

    def execute_query(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a query synchronously and return the fetched results.

        Args:
            query: SQL query string to execute.
            params: Optional sequence of query parameters.

        Returns:
            List of fetched result tuples.

        Raises:
            SQLSafetyViolationError: If query violates read-only mode.
            DatabaseExecutionError: If query execution fails.

        """
        self._validate_safety(query)
        return self._apply_row_cap(self.adapter.execute_query(query, params))

    async def execute_query_async(
        self,
        query: str,
        params: Sequence[JSONPrimitive] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Asynchronously execute a query and return the fetched results.

        Args:
            query: SQL query string to execute.
            params: Optional sequence of query parameters.

        Returns:
            List of fetched result tuples.

        Raises:
            SQLSafetyViolationError: If query violates read-only mode.
            DatabaseExecutionError: If query execution fails.

        """
        self._validate_safety(query)
        return self._apply_row_cap(await self.adapter.execute_query_async(query, params))

    def has_top_level_order_by(self, query: str) -> bool:
        """Detect presence of a top-level `ORDER BY` clause outside parentheses.

        Args:
            query: SQL query string.

        Returns:
            True if a top-level `ORDER BY` clause is present.

        """
        return _has_top_level_order_by_lexer(query)

    def normalize_row(self, row: tuple[JSONPrimitive | decimal.Decimal, ...]) -> tuple[JSONPrimitive, ...]:
        """Normalize row values to handle float/Decimal conversion and `NULL`/`NaN` equivalents.

        Args:
            row: A tuple of raw database column values.

        Returns:
            A normalized tuple suitable for equality and multiset comparison.

        """
        normalized: list[JSONPrimitive] = []
        for raw_val in row:
            val: JSONPrimitive = float(raw_val) if isinstance(raw_val, decimal.Decimal) else raw_val
            if val is None or (isinstance(val, float) and math.isnan(val)):
                normalized.append(None)
            elif isinstance(val, float):
                if val == 0.0:
                    normalized.append(0.0)
                elif math.isinf(val):
                    normalized.append(val)
                else:
                    normalized.append(round(val, -math.floor(math.log10(abs(val))) + 4))
            else:
                normalized.append(val)
        return tuple(normalized)

    def _tag_raw_row(
        self, row: tuple[JSONPrimitive | decimal.Decimal, ...], *, bucket_floats: bool
    ) -> tuple[tuple[str, JSONPrimitive], ...]:
        """Tag each cell in `row` with its canonical type category (`null`, `bool`, `num`, `str`).

        Args:
            row: Raw database result row.
            bucket_floats: Whether to apply significant-digit rounding for O(N) multiset hashing.

        Returns:
            Tuple of `(type_tag, value)` pairs.

        """
        source = self.normalize_row(row) if bucket_floats else row
        tagged: list[tuple[str, JSONPrimitive]] = []
        for raw_cell in source:
            cell: JSONPrimitive = float(raw_cell) if isinstance(raw_cell, decimal.Decimal) else raw_cell
            if cell is None or (isinstance(cell, float) and math.isnan(cell)):
                tagged.append(("null", None))
            elif isinstance(cell, bool):
                tagged.append(("bool", cell))
            elif isinstance(cell, (int, float)):
                tagged.append(("num", 0.0 if float(cell) == 0.0 else float(cell)))
            else:
                tagged.append(("str", str(cell)))
        return tuple(tagged)

    def compare_results(
        self,
        pred_results: Sequence[tuple[JSONPrimitive, ...]],
        truth_results: Sequence[tuple[JSONPrimitive, ...]],
        strict_order: bool,
    ) -> bool:
        """Compare two sets of database result rows using relative/absolute float tolerance.

        Args:
            pred_results: Result rows from the predicted SQL query.
            truth_results: Result rows from the ground-truth SQL query.
            strict_order: Whether row ordering must match exactly.

        Returns:
            True if the result sets match under `self.float_tol` and `strict_order`.

        """
        if len(pred_results) != len(truth_results):
            return False

        raw_pred = [self._tag_raw_row(r, bucket_floats=False) for r in pred_results]
        raw_truth = [self._tag_raw_row(r, bucket_floats=False) for r in truth_results]

        if strict_order:
            return all(_rows_equivalent(rp, rt, self.float_tol) for rp, rt in zip(raw_pred, raw_truth, strict=True))

        bucket_pred = [self._tag_raw_row(r, bucket_floats=True) for r in pred_results]
        bucket_truth = [self._tag_raw_row(r, bucket_floats=True) for r in truth_results]
        if collections.Counter(bucket_pred) == collections.Counter(bucket_truth):
            return True

        # Fallback tolerance matching for floats straddling rounding boundaries (e.g. 0.00004999 vs 0.00005001)
        unmatched = list(raw_truth)
        for rp in raw_pred:
            matched_idx = -1
            for idx, rt in enumerate(unmatched):
                if _rows_equivalent(rp, rt, self.float_tol):
                    matched_idx = idx
                    break
            if matched_idx == -1:
                return False
            unmatched.pop(matched_idx)
        return True

    def compare_queries(self, predicted_sql: str, ground_truth_sql: str) -> bool:
        """Compare the execution results of two SQL queries synchronously.

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
        await self.adapter.close_async()

    def close(self) -> None:
        """Close the underlying database connection."""
        self.adapter.close()

    def __enter__(self) -> Self:
        """Enter synchronous context manager.

        Returns:
            `self` engine instance.

        """
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit synchronous context manager and close database connection.

        Args:
            _exc_type: Exception class if raised within `with` block.
            _exc_val: Exception instance if raised within `with` block.
            _exc_tb: Traceback if raised within `with` block.

        """
        self.close()

    async def __aenter__(self) -> Self:
        """Enter asynchronous context manager.

        Returns:
            `self` engine instance.

        """
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit asynchronous context manager and close database connection.

        Args:
            _exc_type: Exception class if raised within `async with` block.
            _exc_val: Exception instance if raised within `async with` block.
            _exc_tb: Traceback if raised within `async with` block.

        """
        await self.close_async()


__all__ = ["DEFAULT_FLOAT_TOLERANCE", "DEFAULT_MAX_COMPARE_ROWS", "LiveDatabaseEngine"]
