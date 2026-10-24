"""Database testing helpers and fixture generators."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import Any


def get_sample_ddl(dialect: str = "sqlite") -> str:
    """Return a standard sample DDL schema string for testing.

    Args:
        dialect: SQL dialect ('sqlite', 'duckdb', 'postgres').

    Returns:
        SQL DDL string creating 'users' and 'orders' tables.
    """
    if dialect == "postgres":
        return """CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(255), email VARCHAR(255), active BOOLEAN);
CREATE TABLE orders (id SERIAL PRIMARY KEY, user_id INTEGER, amount DECIMAL(10, 2), created_at TIMESTAMP);"""

    if dialect == "duckdb":
        return """CREATE TABLE users (id UINTEGER PRIMARY KEY, name VARCHAR, email VARCHAR, active BOOLEAN);
CREATE TABLE orders (id UINTEGER PRIMARY KEY, user_id UINTEGER, amount DOUBLE, created_at TIMESTAMP);"""

    # SQLite
    return """CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, active INTEGER);
CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, created_at TEXT);"""


def create_in_memory_sqlite(ddl: str = "") -> sqlite3.Connection:
    """Create an in-memory SQLite connection initialized with DDL.

    Args:
        ddl: SQL DDL string to initialize database schema.

    Returns:
        An open sqlite3.Connection instance with initialized tables.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(ddl)
    return conn


def insert_sample_users(conn: sqlite3.Connection, users: Sequence[tuple[int, str, str]]) -> None:
    """Insert sample user rows into the SQLite database.

    Args:
        conn: Open SQLite database connection.
        users: Sequence of (id, name, email) tuples.
    """
    cur = conn.cursor()
    cur.executemany("INSERT INTO users (id, name, email) VALUES (?, ?, ?);", users)
    conn.commit()
    cur.close()


def seed_table(conn: Any, table_name: str, columns: Sequence[str], data: Sequence[Sequence[Any]]) -> None:
    """Insert parameterized rows into a table.

    Args:
        conn: Open database connection.
        table_name: The name of the table to insert into.
        columns: Sequence of column names.
        data: Sequence of row tuples/lists with parameterized data values.
    """
    cur = conn.cursor()
    cols = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))
    query = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders});"
    cur.executemany(query, data)
    conn.commit()
    cur.close()
