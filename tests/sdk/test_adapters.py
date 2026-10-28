"""Unit tests for the base DatabaseAdapter class."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gemma_4_sql.exceptions import DatabaseExecutionError
from gemma_4_sql.sdk.adapters.base import DatabaseAdapter
from gemma_4_sql.type_hints import JSONPrimitive


class DummyAdapter(DatabaseAdapter[MagicMock, MagicMock]):
    """Concrete DatabaseAdapter subclass for unit testing base behavior."""

    def connect(self) -> MagicMock:
        """Create a mock connection object.

        Returns:
            A `MagicMock` connection.

        """
        return MagicMock()

    @property
    def error_classes(self) -> tuple[type[Exception], ...]:
        """Provide handled exception classes for DummyAdapter.

        Returns:
            Tuple containing `ValueError`.

        """
        return (ValueError,)


@pytest.mark.asyncio
async def test_base_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify synchronous and asynchronous execution, feedback, and error propagation on DatabaseAdapter."""
    adapter = DummyAdapter("path", {})

    await adapter.connect_async()

    adapter.conn.cursor = None
    del adapter.conn.cursor

    adapter.conn.execute.return_value.fetchall.return_value = [("row",)]
    res = adapter.execute_with_feedback("SELECT")
    assert res == (True, [("row",)], None)

    res_async = await adapter.execute_with_feedback_async("SELECT")
    assert res_async == (True, [("row",)], None)

    res_q = adapter.execute_query("SELECT")
    assert res_q == [("row",)]

    res_q_async = await adapter.execute_query_async("SELECT")
    assert res_q_async == [("row",)]

    del adapter.conn.close
    adapter.close()

    class CursorMock:
        """Cursor mock without a close method."""

        def __init__(self) -> None:
            """Initialize cursor with a non-null description."""
            self.description = True

        def execute(self, q: str, p: tuple[JSONPrimitive, ...]) -> None:
            """Execute query no-op."""

        def fetchall(self) -> list[tuple[str, ...]]:
            """Return a single mocked row."""
            return [("c_row",)]

    class ConnMock:
        """Connection mock returning CursorMock."""

        def cursor(self) -> CursorMock:
            """Return a new CursorMock."""
            return CursorMock()

        def close(self) -> None:
            """Close connection no-op."""

    monkeypatch.setattr(adapter, "conn", ConnMock())
    res2 = adapter.execute_with_feedback("SELECT")
    assert res2 == (True, [("c_row",)], None)

    res3 = adapter.execute_query("SELECT")
    assert res3 == [("c_row",)]

    adapter.close()
    adapter.setup_schema("DDL")

    adapter.conn = MagicMock()
    adapter.conn.cursor.return_value.execute.side_effect = ValueError("err")
    res_err = adapter.execute_with_feedback("SELECT")
    assert res_err == (False, [], "err")

    with pytest.raises(DatabaseExecutionError, match="err"):
        adapter.execute_query("SELECT")

    adapter.conn.cursor.return_value.execute.side_effect = None
    adapter.conn.cursor.return_value.close.side_effect = ValueError("err_close")
    with pytest.raises(DatabaseExecutionError, match="err_close"):
        adapter.execute_query("SELECT")

    class CursorMockClose:
        """Cursor mock with a close method."""

        def __init__(self) -> None:
            """Initialize cursor with a non-null description."""
            self.description = True

        def execute(self, q: str, p: tuple[JSONPrimitive, ...]) -> None:
            """Execute query no-op."""

        def fetchall(self) -> list[tuple[str, ...]]:
            """Return a single mocked row."""
            return [("c_row",)]

        def close(self) -> None:
            """Close cursor no-op."""

    class ConnMockClose:
        """Connection mock returning CursorMockClose."""

        def cursor(self) -> CursorMockClose:
            """Return a new CursorMockClose."""
            return CursorMockClose()

        def close(self) -> None:
            """Close connection no-op."""

    monkeypatch.setattr(adapter, "conn", ConnMockClose())
    assert adapter.execute_with_feedback("SELECT") == (True, [("c_row",)], None)
    assert adapter.execute_query("SELECT") == [("c_row",)]
