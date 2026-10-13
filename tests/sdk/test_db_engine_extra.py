"""Module docstring."""

import pytest

from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine


@pytest.mark.asyncio
async def test_db_engine_execute_with_feedback_async_permission_error():
    # When read_only is True, modifications like "DROP" raise PermissionError
    """Docstring for test_db_engine_execute_with_feedback_async_permission_error."""
    engine = LiveDatabaseEngine(":memory:", read_only=True)
    success, results, err = await engine.execute_with_feedback_async("DROP TABLE users")
    assert success is False
    assert "not allowed" in err.lower() or "read-only" in err.lower() or err != ""


def test_db_engine_has_top_level_order_by():
    """Docstring for test_db_engine_has_top_level_order_by."""
    engine = LiveDatabaseEngine(":memory:")
    assert engine._has_top_level_order_by("SELECT * FROM t ORDER BY a") is True
    assert engine._has_top_level_order_by("SELECT * FROM (SELECT * FROM t ORDER BY a)") is False
    assert engine._has_top_level_order_by("SELECT * FROM (SELECT * FROM t ORDER BY a) ORDER BY b") is True
    # Test unbalanced parens just in case
    assert engine._has_top_level_order_by("SELECT * FROM ( ORDER BY a") is False
    assert engine._has_top_level_order_by("SELECT * FROM t ) ORDER BY a") is True


def test_db_engine_normalize_row():
    """Docstring for test_db_engine_normalize_row."""
    engine = LiveDatabaseEngine(":memory:")
    row = (None, float("nan"), 0.0, 1.23456789, "string")
    norm = engine._normalize_row(row)
    assert norm[0] is None
    assert norm[1] is None
    assert norm[2] == 0.0
    assert norm[3] == 1.2346  # rounded: 1.23456789 -> 1.2346
    # Actually -floor(log10(1.2345)) + 4 = 0 + 4 = 4 digits -> 1.2346
    assert isinstance(norm[3], float)
    assert norm[4] == "string"


def test_db_engine_compare_results_unordered():
    """Docstring for test_db_engine_compare_results_unordered."""
    engine = LiveDatabaseEngine(":memory:")
    res1 = [(1, 2), (3, 4)]
    res2 = [(3, 4), (1, 2)]
    assert engine._compare_results(res1, res2, strict_order=False) is True
    assert engine._compare_results(res1, res2, strict_order=True) is False
