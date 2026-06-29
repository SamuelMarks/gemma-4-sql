"""Tests for missing DB Engine coverage."""

from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine


def test_db_engine_insert_no_description() -> object:  # type: ignore[return]
    """Initialize function test_db_engine_insert_no_description."""
    engine = LiveDatabaseEngine(db_path=":memory:", db_type="sqlite")
    engine.conn.execute("CREATE TABLE t (id INT)")
    (success, res, err) = engine.execute_with_feedback("INSERT INTO t VALUES (1)")
    if success is not True:
        raise AssertionError
    if not res == []:
        raise AssertionError
    if err is not None:
        raise AssertionError
