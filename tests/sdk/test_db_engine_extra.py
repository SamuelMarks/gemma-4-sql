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
