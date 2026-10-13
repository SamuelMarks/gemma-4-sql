"""Tests for db engine."""

import pytest

from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine


def test_db_engine_live() -> None:
    """Test LiveDatabaseEngine with real SQLite execution."""
    engine = LiveDatabaseEngine(db_type="sqlite", db_path=":memory:", ddl="CREATE TABLE t (a INT);", read_only=False)
    engine.connect()

    # test query
    res = engine.execute_query("INSERT INTO t VALUES (1)")
    assert res == []

    res = engine.execute_query("SELECT * FROM t")
    assert res == [(1,)]

    # test compare queries
    engine.execute_query("INSERT INTO t VALUES (2)")
    assert engine.compare_queries("SELECT * FROM t WHERE a=1", "SELECT a FROM t WHERE a=1") is True
    assert engine.compare_queries("SELECT * FROM t", "SELECT * FROM t WHERE a=1") is False

    # test compare queries with error
    assert engine.compare_queries("SELECT * FROM t", "BAD SQL") is False
    assert engine.compare_queries("BAD SQL", "BAD SQL2") is True

    engine.close()


def test_db_engine_unsupported() -> None:
    """Test unsupported type."""
    with pytest.raises(ValueError, match="Unsupported db_type"):
        LiveDatabaseEngine(db_type="invalid_type")


def test_db_engine_safety() -> None:
    """Test _validate_safety against malicious SQL."""
    engine = LiveDatabaseEngine(db_type="sqlite", db_path=":memory:", read_only=True)
    engine.setup_schema("CREATE TABLE t (id INT)")  # setup schema

    # test safe query
    suc, res, err = engine.execute_with_feedback("SELECT * FROM t")
    assert suc is True

    # test unsafe query
    unsafe_queries = ["DROP TABLE t", "TRUNCATE t", "ALTER TABLE t ADD COLUMN b INT", "GRANT ALL ON t TO user", "INSERT INTO t VALUES (1)", "UPDATE t SET id=2", "DELETE FROM t", "SELECT * FROM t; DROP TABLE t", "SELECT * FROM t /* comment */ DROP TABLE t"]

    for q in unsafe_queries:
        suc, res, err = engine.execute_with_feedback(q)
        assert suc is False
        assert "Safety Violation" in str(err)

        with pytest.raises(PermissionError):
            engine.execute_query(q)


@pytest.mark.asyncio
async def test_db_engine_async() -> None:
    """Test async methods."""
    engine = LiveDatabaseEngine(db_type="sqlite", db_path=":memory:", read_only=False)
    engine.setup_schema("CREATE TABLE t (a INT)")

    await engine.execute_query_async("INSERT INTO t VALUES (1)")

    res = await engine.execute_query_async("SELECT * FROM t")
    assert res == [(1,)]

    suc, res, err = await engine.execute_with_feedback_async("SELECT * FROM t")
    assert suc is True
    assert res == [(1,)]

    assert await engine.compare_queries_async("SELECT * FROM t", "SELECT a FROM t") is True
