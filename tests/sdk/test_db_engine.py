"""Unit tests for LiveDatabaseEngine SQL execution, comparison, and read-only safety guards."""

from __future__ import annotations

import pytest

from gemma_4_sql.exceptions import ConfigurationError, SQLSafetyViolationError
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine


def test_db_engine_live() -> None:
    """Test LiveDatabaseEngine with real SQLite execution and query comparison."""
    engine = LiveDatabaseEngine(db_type="sqlite", db_path=":memory:", ddl="CREATE TABLE t (a INT);", read_only=False)
    engine.connect()

    res = engine.execute_query("INSERT INTO t VALUES (1)")
    assert res == []

    res = engine.execute_query("SELECT * FROM t")
    assert res == [(1,)]

    engine.execute_query("INSERT INTO t VALUES (2)")
    assert engine.compare_queries("SELECT * FROM t WHERE a=1", "SELECT a FROM t WHERE a=1") is True
    assert engine.compare_queries("SELECT * FROM t", "SELECT * FROM t WHERE a=1") is False

    # If either or both queries are invalid SQL, compare_queries must return False
    assert engine.compare_queries("SELECT * FROM t", "BAD SQL") is False
    assert engine.compare_queries("BAD SQL", "SELECT * FROM t") is False
    assert engine.compare_queries("BAD SQL", "BAD SQL2") is False

    engine.close()


def test_db_engine_unsupported() -> None:
    """Verify unsupported db_type raises ConfigurationError (a ValueError subclass)."""
    with pytest.raises(ConfigurationError, match="Unsupported db_type"):
        LiveDatabaseEngine(db_type="invalid_type")


def test_db_engine_safety() -> None:
    """Verify _validate_safety blocks mutating and stacked SQL while allowing literals with `--` or `;`."""
    engine = LiveDatabaseEngine(db_type="sqlite", db_path=":memory:", read_only=True)
    engine.setup_schema("CREATE TABLE t (id INT, note TEXT)")

    # Safe queries including semicolons and comment sequences inside string literals
    suc, res, err = engine.execute_with_feedback("SELECT * FROM t")
    assert suc is True
    assert res == []
    assert err is None

    suc_lit, res_lit, err_lit = engine.execute_with_feedback("SELECT '-- not a comment; DROP TABLE t', 'a;b' FROM t")
    assert suc_lit is True
    assert res_lit == []
    assert err_lit is None

    # Dollar-quoted string literals containing semicolons and keywords are allowed in _validate_safety
    engine._validate_safety("SELECT $$DROP TABLE t; SELECT 1$$, $tag$DELETE FROM t$tag$ FROM t")

    unsafe_queries = [
        "DROP TABLE t",
        "TRUNCATE t",
        "ALTER TABLE t ADD COLUMN b INT",
        "GRANT ALL ON t TO user",
        "INSERT INTO t VALUES (1, 'x')",
        "UPDATE t SET id=2",
        "DELETE FROM t",
        "SELECT * FROM t; SELECT * FROM t",
        "SELECT * FROM t;;",
        "SELECT * FROM t; DROP TABLE t",
        "SELECT * FROM t /* comment */ DROP TABLE t",
        "SELECT * FROM t /* unclosed block comment",
    ]

    for q in unsafe_queries:
        suc, _res, err = engine.execute_with_feedback(q)
        assert suc is False
        assert "Safety Violation" in str(err)

        with pytest.raises(SQLSafetyViolationError):
            engine.execute_query(q)


@pytest.mark.asyncio
async def test_db_engine_async() -> None:
    """Test asynchronous connection, deferred DDL, query execution, feedback, and comparison methods."""
    engine = LiveDatabaseEngine(
        db_type="sqlite",
        db_path=":memory:",
        ddl="CREATE TABLE t (a INT);",
        read_only=False,
        defer_ddl=True,
    )
    await engine.connect_async()

    await engine.execute_query_async("INSERT INTO t VALUES (1)")

    res = await engine.execute_query_async("SELECT * FROM t")
    assert res == [(1,)]

    suc, res, err = await engine.execute_with_feedback_async("SELECT * FROM t")
    assert suc is True
    assert res == [(1,)]
    assert err is None

    assert await engine.compare_queries_async("SELECT * FROM t", "SELECT a FROM t") is True
    assert await engine.compare_queries_async("BAD SQL 1", "BAD SQL 2") is False
    engine.close()


@pytest.mark.asyncio
async def test_db_engine_execute_with_feedback_async_permission_error() -> None:
    """Verify execute_with_feedback_async catches SQLSafetyViolationError in read-only mode."""
    engine = LiveDatabaseEngine(":memory:", read_only=True)
    success, results, err = await engine.execute_with_feedback_async("DROP TABLE users")
    assert success is False
    assert results == []
    assert err is not None
    assert "safety violation" in err.lower()
    engine.close()


def test_db_engine_has_top_level_order_by_and_normalization() -> None:
    """Verify _has_top_level_order_by, _normalize_row, and _compare_results edge cases."""
    engine = LiveDatabaseEngine(":memory:")
    assert engine._has_top_level_order_by("SELECT * FROM t ORDER BY a") is True
    assert engine._has_top_level_order_by("SELECT * FROM (SELECT * FROM t ORDER BY a)") is False
    assert engine._has_top_level_order_by("SELECT * FROM (SELECT * FROM t ORDER BY a) ORDER BY b") is True
    assert engine._has_top_level_order_by("SELECT * FROM ( ORDER BY a") is False
    assert engine._has_top_level_order_by("SELECT * FROM t ) ORDER BY a") is True

    row = (None, float("nan"), 0.0, 1.23456789, "string")
    norm = engine._normalize_row(row)
    assert norm[0] is None
    assert norm[1] is None
    assert norm[2] == 0.0
    assert norm[3] == 1.2346
    assert isinstance(norm[3], float)
    assert norm[4] == "string"

    res1 = [(1, 2), (3, 4)]
    res2 = [(3, 4), (1, 2)]
    assert engine._compare_results(res1, res2, strict_order=False) is True
    assert engine._compare_results(res1, res2, strict_order=True) is False
    assert engine._compare_results([(True, None, "a")], [(1, None, "a")], strict_order=True) is False
    assert engine._compare_results([(None,)], [("NULL",)], strict_order=True) is False
    assert engine._compare_results([(True,)], [(True,)], strict_order=True) is True
    engine.close()


@pytest.mark.asyncio
async def test_db_engine_context_managers() -> None:
    """Verify LiveDatabaseEngine supports both `with` and `async with` context management."""
    with LiveDatabaseEngine(":memory:") as sync_engine:
        rows = sync_engine.execute_query("SELECT 1")
        assert rows == [(1,)]

    async with LiveDatabaseEngine(":memory:") as async_engine:
        rows_async = await async_engine.execute_query_async("SELECT 2")
        assert rows_async == [(2,)]

    # Also test close_async when adapter has a coroutine close_async method
    engine_coro = LiveDatabaseEngine(":memory:")
    closed_flag = False

    async def _fake_close_async() -> None:
        """Set closed_flag to True."""
        nonlocal closed_flag
        closed_flag = True

    setattr(engine_coro.adapter, "close_async", _fake_close_async)
    await engine_coro.close_async()
    assert closed_flag is True
