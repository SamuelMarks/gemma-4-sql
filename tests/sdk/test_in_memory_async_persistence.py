"""Tests for in-memory SQLite and DuckDB async persistence and parallel query execution."""

from __future__ import annotations

import asyncio

import pytest

from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine


@pytest.mark.asyncio
async def test_sqlite_in_memory_async_persistence() -> None:
    """Test SQLite :memory: database async persistence and parallel queries.

    Returns:
        None.
    """
    ddl = """
    CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);
    INSERT INTO users VALUES (1, 'Alice');
    INSERT INTO users VALUES (2, 'Bob');
    """
    engine = LiveDatabaseEngine(db_path=":memory:", db_type="sqlite", ddl=ddl)
    await engine.connect_async()

    # Verify rows persisted in async queries
    rows = await engine.execute_query_async("SELECT name FROM users ORDER BY id")
    assert rows == [("Alice",), ("Bob",)]

    # Verify feedback execution
    success, feedback_rows, err = await engine.execute_with_feedback_async("SELECT COUNT(*) FROM users")
    assert success is True
    assert feedback_rows == [(2,)]
    assert err is None

    # Test parallel async execution
    q1 = engine.execute_query_async("SELECT name FROM users WHERE id = 1")
    q2 = engine.execute_query_async("SELECT name FROM users WHERE id = 2")
    res1, res2 = await asyncio.gather(q1, q2)
    assert res1 == [("Alice",)]
    assert res2 == [("Bob",)]


@pytest.mark.asyncio
async def test_duckdb_in_memory_async_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test DuckDB :memory: database async persistence and parallel queries.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    # Ensure real duckdb module is used for DuckDBAdapter
    real_duck = getattr(pytest, "_real_duckdb", None)
    if real_duck is not None:
        monkeypatch.setitem(__import__("sys").modules, "duckdb", real_duck)
        monkeypatch.setattr("gemma_4_sql.sdk.adapters.duckdb_adapter.duckdb", real_duck)

    ddl = """
    CREATE TABLE products (id INT, title TEXT);
    INSERT INTO products VALUES (10, 'Laptop');
    INSERT INTO products VALUES (20, 'Phone');
    """
    engine = LiveDatabaseEngine(db_path=":memory:", db_type="duckdb", ddl=ddl)
    await engine.connect_async()

    rows = await engine.execute_query_async("SELECT title FROM products ORDER BY id")
    assert rows == [("Laptop",), ("Phone",)]

    success, feedback_rows, err = await engine.execute_with_feedback_async("SELECT COUNT(*) FROM products")
    assert success is True
    assert feedback_rows == [(2,)]
    assert err is None

    # Test parallel async queries via asyncio.gather
    t1 = engine.execute_query_async("SELECT title FROM products WHERE id = 10")
    t2 = engine.execute_query_async("SELECT title FROM products WHERE id = 20")
    res1, res2 = await asyncio.gather(t1, t2)
    assert res1 == [("Laptop",)]
    assert res2 == [("Phone",)]


@pytest.mark.asyncio
async def test_sqlite_async_rollback_on_error() -> None:
    """Test SQLite async execution error handling and rollback on bad queries.

    Returns:
        None.
    """
    ddl = "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT);"
    engine = LiveDatabaseEngine(db_path=":memory:", db_type="sqlite", ddl=ddl, read_only=False)
    await engine.connect_async()

    # Insert initial row
    await engine.execute_query_async("INSERT INTO accounts VALUES (1, 100)")

    # Execute a failing query
    success, rows, err = await engine.execute_with_feedback_async("INSERT INTO accounts VALUES (1, 200)")
    assert success is False
    assert rows == []
    assert err is not None

    # Verify state remains consistent
    res = await engine.execute_query_async("SELECT balance FROM accounts WHERE id = 1")
    assert res == [(100,)]


@pytest.mark.asyncio
async def test_sqlite_async_schema_migration() -> None:
    """Test SQLite async schema migration by altering table structure and querying.

    Returns:
        None.
    """
    ddl = "CREATE TABLE items (id INT, name TEXT);"
    engine = LiveDatabaseEngine(db_path=":memory:", db_type="sqlite", ddl=ddl, read_only=False)
    await engine.connect_async()

    await engine.execute_query_async("INSERT INTO items VALUES (1, 'hammer')")
    # Schema migration: ADD COLUMN
    await engine.execute_query_async("ALTER TABLE items ADD COLUMN price INT DEFAULT 0")
    await engine.execute_query_async("UPDATE items SET price = 25 WHERE id = 1")

    rows = await engine.execute_query_async("SELECT name, price FROM items WHERE id = 1")
    assert rows == [("hammer", 25)]
