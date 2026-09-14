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
