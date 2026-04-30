"""
Live Database Evaluation Engine for Text-to-SQL execution accuracy.
"""

from __future__ import annotations

import sqlite3
from typing import Any

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
except ImportError:  # pragma: no cover
    duckdb = None  # pragma: no cover

class LiveDatabaseEngine:
    """
    Executes SQL queries against an underlying database engine
    to measure execution accuracy (EX) of generated Text-to-SQL queries.
    Supports SQLite, PostgreSQL, Snowflake, and DuckDB.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        ddl: str | None = None,
        db_type: str = "sqlite",
        db_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initializes the LiveDatabaseEngine.

        Args:
            db_path: Path to the database or connection URI. Defaults to an in-memory DB for sqlite and duckdb.
            ddl: Optional SQL Data Definition Language string to initialize the schema.
            db_type: The type of database backend ('sqlite', 'postgresql', 'snowflake', 'duckdb').
            db_kwargs: Additional keyword arguments for the database connection.
        """
        self.db_path = db_path
        self.db_type = db_type.lower()
        self.db_kwargs = db_kwargs or {}
        self.conn = self._connect()

        if ddl:
            self.setup_schema(ddl)

    def _connect(self) -> Any:
        """Connect to database."""
        if self.db_type == "sqlite":
            return sqlite3.connect(self.db_path, **self.db_kwargs)
        elif self.db_type == "postgresql":
            if psycopg2 is None:
                raise ImportError(
                    "psycopg2 is required for PostgreSQL support. Install with `pip install psycopg2-binary`."
                )
            if self.db_path and self.db_path != ":memory:":
                return psycopg2.connect(self.db_path, **self.db_kwargs)
            return psycopg2.connect(**self.db_kwargs)
        elif self.db_type == "snowflake":
            if snowflake is None:
                raise ImportError(
                    "snowflake-connector-python is required for Snowflake support."
                )
            return snowflake.connector.connect(**self.db_kwargs)
        elif self.db_type == "duckdb":
            if duckdb is None:
                raise ImportError(
                    "duckdb is required for DuckDB support. Install with `pip install duckdb`."
                )
            return duckdb.connect(self.db_path, **self.db_kwargs)
        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")

    def setup_schema(self, ddl: str) -> None:
        """
        Executes DDL statements to construct the database schema.

        Args:
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

    def execute_with_feedback(
        self, query: str
    ) -> tuple[bool, list[tuple[Any, ...]], str | None]:
        """
        Executes a query and returns execution success status, results, and error message.

        Args:
            query: The SQL query to execute.

        Returns:
            A tuple of (success, results, error_message).
        """
        try:
            if self.db_type == "duckdb":
                results = self.conn.execute(query).fetchall()
                return True, results, None
                
            cursor = self.conn.cursor()
            cursor.execute(query)
            if cursor.description is not None:
                return True, cursor.fetchall(), None
            return True, [], None
        except Exception as e:
            return False, [], str(e)
        finally:
            if self.db_type != "duckdb" and "cursor" in locals():
                cursor.close()

    def execute_query(self, query: str) -> list[tuple[Any, ...]]:
        """
        Executes a query and returns the fetched results.

        Args:
            query: The SQL query to execute.

        Returns:
            A list of tuples containing the result rows. Returns an empty list
            if the query fails due to a syntax or execution error.
        """
        try:
            if self.db_type == "duckdb":
                return self.conn.execute(query).fetchall()
                
            cursor = self.conn.cursor()
            cursor.execute(query)
            if cursor.description is not None:
                return cursor.fetchall()
            return []
        except Exception:
            return []
        finally:
            if self.db_type != "duckdb" and "cursor" in locals():
                cursor.close()

    def compare_queries(self, predicted_sql: str, ground_truth_sql: str) -> bool:
        """
        Compares the execution results of two queries.

        Args:
            predicted_sql: The SQL query generated by the model.
            ground_truth_sql: The expected SQL query.

        Returns:
            True if both queries return the identical result set, False otherwise.
        """
        pred_results = self.execute_query(predicted_sql)
        truth_results = self.execute_query(ground_truth_sql)
        return pred_results == truth_results

    def close(self) -> None:
        """Closes the database connection."""
        self.conn.close()
