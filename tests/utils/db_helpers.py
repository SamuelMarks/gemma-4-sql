"""Database testing helpers and fixture generators."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence


def get_sample_ddl() -> str:
    """Return a standard sample DDL schema string for testing.

    Returns:
        SQL DDL string creating 'users' and 'orders' tables.
    """
    return """CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL);"""


def create_in_memory_sqlite(ddl: str | None = None) -> sqlite3.Connection:
    """Create an in-memory SQLite connection initialized with optional DDL.

    Args:
        ddl: Optional SQL DDL string to initialize database schema.

    Returns:
        An open sqlite3.Connection instance with initialized tables.
    """
    conn = sqlite3.connect(":memory:")
    if ddl:
        conn.executescript(ddl)
    return conn


def insert_sample_users(conn: sqlite3.Connection, users: Sequence[tuple[int, str, str]]) -> None:
    """Insert sample user rows into the SQLite database.

    Args:
        conn: Open SQLite database connection.
        users: Sequence of (id, name, email) tuples.
    """
    cur = conn.cursor()
    cur.executemany("INSERT INTO users VALUES (?, ?, ?);", users)
    conn.commit()
    cur.close()
