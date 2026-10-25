"""Base adapter protocol for database connections."""

from __future__ import annotations

import abc
import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

from gemma_4_sql.exceptions import DatabaseExecutionError, SQLSafetyViolationError

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONPrimitive

logger = logging.getLogger(__name__)

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
    (\$(?P<dtag>[A-Za-z0-9_]*)\$[\s\S]*?\$(?P=dtag)\$) | # PostgreSQL dollar-quoted string literal
    ('(?:''|\\.|[^'\\])*')      | # Single-quoted string literal (SQL & backslash escapes)
    ("(?:""|\\.|[^"\\])*")      | # Double-quoted identifier/literal
    (`(?:``|[^`])*`)            | # Backtick-quoted identifier
    (\[[^\]]*\])                | # Bracket-quoted identifier
    (/\*[\s\S]*?\*/)            | # Block comment
    (--[^\n]*)                    # Single-line comment
    """,
    re.VERBOSE,
)


_SQL_COMMENT_PATTERN = re.compile(
    r"""
    ('(?:''|\\.|[^'\\])*')      | # Single-quoted string literal
    ("(?:""|\\.|[^"\\])*")      | # Double-quoted identifier/literal
    (`(?:``|[^`])*`)            | # Backtick-quoted identifier
    (\[[^\]]*\])                | # Bracket-quoted identifier
    (?P<comment>/\*[\s\S]*?\*/|--[^\n]*) # Block or single-line comment
    """,
    re.VERBOSE,
)


def strip_sql_comments(query: str) -> str:
    """Strip SQL block and single-line comments while preserving quoted literals and identifiers.

    Args:
        query: Raw SQL query or DDL string.

    Returns:
        SQL string with comments replaced by spaces and literals preserved intact.

    """
    return _SQL_COMMENT_PATTERN.sub(lambda m: " " if m.group("comment") is not None else m.group(0), query)


def strip_comments_and_literals(query: str) -> str:
    """Replace SQL comments and quoted literals with spaces in a single lexical pass.

    Args:
        query: Raw SQL query string.

    Returns:
        SQL string with all comments and quoted segments replaced by a single space.

    """
    return _SQL_TOKEN_PATTERN.sub(" ", query)


def validate_readonly_sql(query: str, *, read_only: bool = True) -> None:
    """Ensure the SQL query is safe to execute when `read_only` is True.

    Args:
        query: SQL query string to inspect.
        read_only: Whether read-only enforcement is active.

    Raises:
        SQLSafetyViolationError: If mutating statements or stacked queries are attempted in read-only mode.

    """
    if not read_only:
        return

    cleaned = strip_comments_and_literals(query)
    if "/*" in cleaned:
        msg = "Safety Violation: Unclosed block comment is not allowed in read-only mode."
        raise SQLSafetyViolationError(msg)

    statements = [stmt.strip() for stmt in cleaned.split(";") if stmt.strip()]
    if len(statements) > 1 or cleaned.count(";") > 1:
        msg = "Safety Violation: Multiple stacked SQL statements are not allowed in read-only mode."
        raise SQLSafetyViolationError(msg)

    words = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", cleaned.upper())
    for word in words:
        if word in _DANGEROUS_KEYWORDS:
            msg = f"Safety Violation: Mutating statements ({word}) are not allowed in read-only mode."
            raise SQLSafetyViolationError(msg)


class DatabaseAdapter(abc.ABC):
    """Abstract base class for database dialect adapters."""

    def __init__(self, db_path: str, db_kwargs: dict[str, object], *, read_only: bool = False) -> None:
        """Initialize the database adapter and establish initial connection.

        Args:
            db_path: The file path or connection URI for the database.
            db_kwargs: Additional connection keyword arguments.
            read_only: Flag indicating whether mutating queries are disallowed.

        """
        self.db_path = db_path
        self.db_kwargs = db_kwargs
        self.read_only = read_only
        self._owns_connection: bool = not (("existing_conn" in db_kwargs and db_kwargs["existing_conn"] is not None) or ("conn" in db_kwargs and db_kwargs["conn"] is not None))
        self.conn: Any = self.connect()

    @property
    @abc.abstractmethod
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Provide the tuple of database-specific exception types caught by this adapter."""

    @abc.abstractmethod
    def connect(self) -> object:
        """Connect synchronously to the underlying database.

        Returns:
            The active database connection object.

        """

    async def connect_async(self) -> object:
        """Connect asynchronously to the underlying database.

        Returns:
            The active asynchronous database connection object.

        """
        return await asyncio.to_thread(self.connect)

    def setup_schema(self, ddl: str) -> None:
        """Execute DDL statements to initialize the database schema.

        Args:
            ddl: The SQL DDL query string.

        """
        self.execute_with_feedback(ddl)

    def execute_with_feedback(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute a query synchronously and return execution status, rows, and error feedback.

        Args:
            query: The SQL query string to execute.
            params: Optional positional parameters for the query.

        Returns:
            A tuple containing `(success, rows, error_message)`.

        """
        try:
            rows = self.execute_query(query, params)
        except self.error_classes + (DatabaseExecutionError,) as e:
            cause = e.cause if isinstance(e, DatabaseExecutionError) and e.cause is not None else e
            return (False, [], str(cause))
        else:
            return (True, rows, None)

    async def execute_with_feedback_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Execute a query asynchronously and return execution status, rows, and error feedback.

        Args:
            query: The SQL query string to execute.
            params: Optional positional parameters for the query.

        Returns:
            A tuple containing `(success, rows, error_message)`.

        """
        return await asyncio.to_thread(self.execute_with_feedback, query, params)

    def execute_query(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a query synchronously and return fetched result tuples.

        Args:
            query: The SQL query string to execute.
            params: Optional positional parameters for the query.

        Returns:
            List of result row tuples.

        Raises:
            DatabaseExecutionError: If query execution fails on the underlying connection.

        """
        try:
            if hasattr(self.conn, "cursor"):
                cursor = self.conn.cursor()
                cursor.execute(query, params or ())
                results: list[tuple[JSONPrimitive, ...]] = [tuple(r) for r in cursor.fetchall()] if getattr(cursor, "description", None) is not None else []
                if hasattr(cursor, "close"):
                    cursor.close()
                return results
            raw_rows: Any = self.conn.execute(query, params or ()).fetchall()
            return [tuple(r) for r in raw_rows]
        except self.error_classes as e:
            logger.debug("Query execution failed: %s", e)
            raise DatabaseExecutionError(str(e), cause=e) from e

    async def execute_query_async(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> list[tuple[JSONPrimitive, ...]]:
        """Execute a query asynchronously and return fetched result tuples.

        Args:
            query: The SQL query string to execute.
            params: Optional positional parameters for the query.

        Returns:
            List of result row tuples.

        Raises:
            DatabaseExecutionError: If query execution fails on the underlying connection.

        """
        return await asyncio.to_thread(self.execute_query, query, params)

    def close(self) -> None:
        """Close the active database connection if owned by this adapter and open."""
        if self._owns_connection and hasattr(self.conn, "close"):
            self.conn.close()
