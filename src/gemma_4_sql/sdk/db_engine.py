"""Live Database Evaluation Engine for Text-to-SQL execution accuracy."""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import typing

logger = logging.getLogger(__name__)

try:
    import psycopg2
except ImportError:  # pragma: no cover
    psycopg2 = None
try:
    import snowflake.connector
except ImportError:  # pragma: no cover
    snowflake = None
try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None
try:
    import aiosqlite
except ImportError:  # pragma: no cover
    aiosqlite = None
try:
    import asyncpg
except ImportError:  # pragma: no cover
    asyncpg = None


class LiveDatabaseEngine:
    """Execute SQL queries against an underlying database engine.

    to measure execution accuracy (EX) of generated Text-to-SQL queries.
    Supports SQLite, PostgreSQL, Snowflake, and DuckDB.
    """

    def __init__(self: typing.Any, db_path: str = ":memory:", ddl: str | None = None, db_type: str = "sqlite", db_kwargs: dict[str, object] | None = None, read_only: bool = True) -> None:
        """Initialize the LiveDatabaseEngine.

        Args:
        ----
            db_path: Path to the database or connection URI. Defaults to an in-memory DB for sqlite and duckdb.
            ddl: Optional SQL Data Definition Language string to initialize the schema.
            db_type: The type of database backend ('sqlite', 'postgresql', 'snowflake', 'duckdb').
            db_kwargs: Additional keyword arguments for the database connection.
            read_only: If True, prevents execution of mutating statements (INSERT, UPDATE, DELETE, DROP).

        """
        self.db_path = db_path
        self.db_type = db_type.lower()
        self.db_kwargs = db_kwargs or {}
        self.read_only = read_only
        self.conn = self.connect()
        if ddl:
            # We temporarily bypass read-only to setup schema
            old_ro = self.read_only
            self.read_only = False
            try:
                self.setup_schema(ddl)
            finally:
                self.read_only = old_ro

    def connect(self: typing.Any) -> object:
        """Connect to database.

        Returns
        -------
            A database connection object specific to the db_type.

        """
        if self.db_type == "sqlite":
            return sqlite3.connect(self.db_path, **self.db_kwargs)
        if self.db_type == "postgresql":
            if psycopg2 is None:
                msg = "psycopg2 is required for PostgreSQL support. Install with `pip install psycopg2-binary`."
                raise ImportError(msg)
            if self.db_path and self.db_path != ":memory:":
                return psycopg2.connect(self.db_path, **self.db_kwargs)
            return psycopg2.connect(**self.db_kwargs)
        if self.db_type == "snowflake":
            if snowflake is None:
                msg = "snowflake-connector-python is required for Snowflake support."
                raise ImportError(msg)
            return snowflake.connector.connect(**self.db_kwargs)
        if self.db_type == "duckdb":
            if duckdb is None:
                msg = "duckdb is required for DuckDB support. Install with `pip install duckdb`."
                raise ImportError(msg)

            # DuckDB natively supports read_only flag
            kwargs = self.db_kwargs.copy()
            if self.read_only and self.db_path != ":memory:":
                kwargs["read_only"] = True
            return duckdb.connect(self.db_path, **kwargs)

        msg = f"Unsupported db_type: {self.db_type}"
        raise ValueError(msg)

    async def connect_async(self: typing.Any) -> object:
        """Asynchronously connect to database.

        Returns
        -------
            An asynchronous database connection object.

        """
        if self.db_type == "sqlite":
            if aiosqlite is None:
                msg = "aiosqlite is required for async SQLite support."
                raise ImportError(msg)
            return await aiosqlite.connect(self.db_path, **self.db_kwargs)
        if self.db_type == "postgresql":
            if asyncpg is None:
                msg = "asyncpg is required for async PostgreSQL support."
                raise ImportError(msg)
            if self.db_path and self.db_path != ":memory:":
                return await asyncpg.connect(self.db_path, **self.db_kwargs)
            return await asyncpg.connect(**self.db_kwargs)
        if self.db_type == "duckdb":
            # DuckDB is synchronous and doesn't natively support asyncio connections.
            # We return the synchronous connection and wrap operations in thread pool
            return self.conn

        msg = f"Async operations not natively supported for db_type: {self.db_type}"
        raise ValueError(msg)

    def _validate_safety(self: typing.Any, query: str) -> None:
        """Ensure the query is safe to execute if read_only is True."""
        if not self.read_only:
            return

        # Basic heuristic sandbox protection against mutating statements
        dangerous_patterns = [r"\bDROP\b", r"\bDELETE\b", r"\bUPDATE\b", r"\bINSERT\b", r"\bALTER\b", r"\bTRUNCATE\b", r"\bGRANT\b", r"\bREVOKE\b"]
        upper_query = query.upper()
        for pattern in dangerous_patterns:
            if re.search(pattern, upper_query):
                msg = f"Safety Violation: Mutating statements ({pattern}) are not allowed in read-only mode."
                raise PermissionError(msg)

    def setup_schema(self: typing.Any, ddl: str) -> None:
        """Execute DDL statements to construct the database schema.

        Args:
        ----
            ddl: The SQL Data Definition Language string.

        """
        self._validate_safety(ddl)
        if self.db_type == "sqlite":
            with self.conn:
                self.conn.executescript(ddl)
        elif self.db_type == "duckdb":
            self.conn.execute(ddl)
        else:
            cursor = self.conn.cursor()
            try:
                cursor.execute(ddl)
                self.conn.commit()
            finally:
                cursor.close()

    def execute_with_feedback(self: typing.Any, query: str) -> tuple[bool, list[tuple[object, ...]], str | None]:
        """Execute a query and returns execution success status, results, and error message.

        Args:
        ----
            query: The SQL query to execute.

        Returns:
        -------
            A tuple of (success, results, error_message).

        """
        try:
            self._validate_safety(query)
            if self.db_type == "duckdb":
                results = self.conn.execute(query).fetchall()
                return (True, results, None)
            cursor = self.conn.cursor()
            cursor.execute(query)
        except Exception as e:
            return (False, [], str(e))
        else:
            if cursor.description is not None:
                return (True, cursor.fetchall(), None)
            return (True, [], None)
        finally:
            if self.db_type != "duckdb" and "cursor" in locals():
                cursor.close()

    async def execute_with_feedback_async(self: typing.Any, query: str) -> tuple[bool, list[tuple[object, ...]], str | None]:
        """Asynchronously execute a query and returns execution success status, results, and error message.

        Args:
        ----
            query: The SQL query to execute.

        Returns:
        -------
            A tuple of (success, results, error_message).

        """
        try:
            self._validate_safety(query)
            if self.db_type == "duckdb":
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(None, lambda: self.conn.execute(query).fetchall())
                return (True, results, None)

            async_conn = await self.connect_async()
            try:
                if self.db_type == "sqlite":
                    cursor = await async_conn.execute(query)
                    try:
                        if cursor.description is not None:
                            results = await cursor.fetchall()
                            return (True, results, None)
                        return (True, [], None)
                    finally:
                        await cursor.close()
                elif self.db_type == "postgresql":
                    # asyncpg
                    records = await async_conn.fetch(query)
                    results = [tuple(r.values()) for r in records]
                    return (True, results, None)
            finally:
                if hasattr(async_conn, "close"):
                    await async_conn.close()
        except Exception as e:
            return (False, [], str(e))

    def execute_query(self: typing.Any, query: str) -> list[tuple[object, ...]]:
        """Execute a query and returns the fetched results.

        Args:
        ----
            query: The SQL query to execute.

        Returns:
        -------
            A list of tuples containing the result rows. Returns an empty list
            if the query fails due to a syntax or execution error.

        """
        try:
            self._validate_safety(query)
            if self.db_type == "duckdb":
                return self.conn.execute(query).fetchall()
            cursor = self.conn.cursor()
            cursor.execute(query)
        except Exception as e:
            logger.debug("Query execution failed: %s", e)
            return []  # pragma: no cover
        else:
            if cursor.description is not None:
                return cursor.fetchall()
            return []  # pragma: no cover
        finally:
            if self.db_type != "duckdb" and "cursor" in locals():
                cursor.close()

    async def execute_query_async(self: typing.Any, query: str) -> list[tuple[object, ...]]:
        """Asynchronously execute a query and returns the fetched results.

        Args:
        ----
            query: The SQL query to execute.

        Returns:
        -------
            A list of tuples containing the result rows. Returns an empty list
            if the query fails due to a syntax or execution error.

        """
        try:
            self._validate_safety(query)
            if self.db_type == "duckdb":
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: self.conn.execute(query).fetchall())

            async_conn = await self.connect_async()
            try:
                if self.db_type == "sqlite":
                    cursor = await async_conn.execute(query)
                    try:
                        if cursor.description is not None:
                            return await cursor.fetchall()
                        return []  # pragma: no cover
                    finally:
                        await cursor.close()
                elif self.db_type == "postgresql":
                    records = await async_conn.fetch(query)
                    return [tuple(r.values()) for r in records]
            finally:
                if hasattr(async_conn, "close"):
                    await async_conn.close()
            return []  # pragma: no cover # noqa: TRY300
        except Exception as e:
            logger.debug("Async Query execution failed: %s", e)
            return []  # pragma: no cover

    def compare_queries(self: typing.Any, predicted_sql: str, ground_truth_sql: str) -> bool:
        """Compare the execution results of two queries.

        Args:
        ----
            predicted_sql: The SQL query generated by the model.
            ground_truth_sql: The expected SQL query.

        Returns:
        -------
            True if both queries return the identical result set, False otherwise.

        """
        pred_results = self.execute_query(predicted_sql)
        truth_results = self.execute_query(ground_truth_sql)
        return pred_results == truth_results  # type: ignore[no-any-return]

    async def compare_queries_async(self: typing.Any, predicted_sql: str, ground_truth_sql: str) -> bool:
        """Asynchronously compare the execution results of two queries.

        Args:
        ----
            predicted_sql: The SQL query generated by the model.
            ground_truth_sql: The expected SQL query.

        Returns:
        -------
            True if both queries return the identical result set, False otherwise.

        """
        pred_results, truth_results = await asyncio.gather(self.execute_query_async(predicted_sql), self.execute_query_async(ground_truth_sql))
        return pred_results == truth_results  # type: ignore[no-any-return]

    def close(self: typing.Any) -> None:
        """Close the database connection."""
        self.conn.close()
