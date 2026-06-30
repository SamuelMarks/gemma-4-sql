"""Live Database Evaluation Engine for Text-to-SQL execution accuracy."""

from __future__ import annotations

import sqlite3
import typing

try:
    import psycopg2
except ImportError:
    psycopg2 = None
try:
    import snowflake.connector
except ImportError:
    snowflake = None
try:
    import duckdb
except ImportError:
    duckdb = None


class LiveDatabaseEngine:
    """Execute SQL queries against an underlying database engine.

    to measure execution accuracy (EX) of generated Text-to-SQL queries.
    Supports SQLite, PostgreSQL, Snowflake, and DuckDB.
    """

    def __init__(self: typing.Any, db_path: str = ":memory:", ddl: str | None = None, db_type: str = "sqlite", db_kwargs: dict[str, object] | None = None) -> None:
        """Initialize the LiveDatabaseEngine.

        Args:
        ----
            db_path: Path to the database or connection URI. Defaults to an in-memory DB for sqlite and duckdb.
            ddl: Optional SQL Data Definition Language string to initialize the schema.
            db_type: The type of database backend ('sqlite', 'postgresql', 'snowflake', 'duckdb').
            db_kwargs: Additional keyword arguments for the database connection.

        """
        self.db_path = db_path
        self.db_type = db_type.lower()
        self.db_kwargs = db_kwargs or {}
        self.conn = self.connect()
        if ddl:
            self.setup_schema(ddl)

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
            return duckdb.connect(self.db_path, **self.db_kwargs)
        msg = f"Unsupported db_type: {self.db_type}"
        raise ValueError(msg)

    def setup_schema(self: typing.Any, ddl: str) -> None:
        """Execute DDL statements to construct the database schema.

        Args:
        ----
            ddl: The SQL Data Definition Language string.

        """
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
            if self.db_type == "duckdb":
                results = self.conn.execute(query).fetchall()
                return (True, results, None)
            cursor = self.conn.cursor()
            cursor.execute(query)
        except RuntimeError as e:
            return (False, [], str(e))
        else:
            if cursor.description is not None:
                return (True, cursor.fetchall(), None)
            return (True, [], None)
        finally:
            if self.db_type != "duckdb" and "cursor" in locals():
                cursor.close()

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
            if self.db_type == "duckdb":
                return self.conn.execute(query).fetchall()
            cursor = self.conn.cursor()
            cursor.execute(query)
        except RuntimeError:
            return []
        else:
            if cursor.description is not None:
                return cursor.fetchall()
            return []
        finally:
            if self.db_type != "duckdb" and "cursor" in locals():
                cursor.close()

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

    def close(self: typing.Any) -> None:
        """Close the database connection."""
        self.conn.close()
