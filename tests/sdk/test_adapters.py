"""Module docstring."""

from unittest.mock import MagicMock

import pytest

from gemma_4_sql.sdk.adapters.base import DatabaseAdapter


class DummyAdapter(DatabaseAdapter):
    """Docstring for DummyAdapter."""

    def connect(self):
        """Docstring for connect."""
        return MagicMock()

    @property
    def error_classes(self):
        """Docstring for error_classes."""
        return (ValueError,)


@pytest.mark.asyncio
async def test_base_adapter():
    """Docstring for test_base_adapter."""
    adapter = DummyAdapter("path", {})

    # Test connect_async
    await adapter.connect_async()

    # Test execute_with_feedback without cursor
    adapter.conn.cursor = None
    del adapter.conn.cursor

    # conn.execute exists
    adapter.conn.execute.return_value.fetchall.return_value = [("row",)]
    res = adapter.execute_with_feedback("SELECT")
    assert res == (True, [("row",)], None)

    # Test execute_with_feedback_async
    res_async = await adapter.execute_with_feedback_async("SELECT")
    assert res_async == (True, [("row",)], None)

    # Test execute_query without cursor
    res_q = adapter.execute_query("SELECT")
    assert res_q == [("row",)]

    # Test execute_query_async
    res_q_async = await adapter.execute_query_async("SELECT")
    assert res_q_async == [("row",)]

    # Test close without conn.close
    del adapter.conn.close
    adapter.close()

    # Test execute_with_feedback with cursor but no cursor.close
    class CursorMock:
        """Docstring for CursorMock."""

        def __init__(self):
            """Docstring for __init__."""
            self.description = True

        def execute(self, q, p):
            """Docstring for execute."""

        def fetchall(self):
            """Docstring for fetchall."""
            return [("c_row",)]

        # no close

    class ConnMock:
        """Docstring for ConnMock."""

        def cursor(self):
            """Docstring for cursor."""
            return CursorMock()

        def close(self):
            """Docstring for close."""

    adapter.conn = ConnMock()
    res2 = adapter.execute_with_feedback("SELECT")
    assert res2 == (True, [("c_row",)], None)

    res3 = adapter.execute_query("SELECT")
    assert res3 == [("c_row",)]

    adapter.close()

    # Test setup_schema
    adapter.setup_schema("DDL")

    # Test execute_with_feedback exception
    adapter.conn = MagicMock()
    adapter.conn.cursor.return_value.execute.side_effect = ValueError("err")
    res = adapter.execute_with_feedback("SELECT")
    assert res == (False, [], "err")

    adapter.conn.cursor.return_value.execute.side_effect = None
    adapter.conn.cursor.return_value.close.side_effect = ValueError("err_close")
    res_q2 = adapter.execute_query("SELECT")
    assert res_q2 == []

    # Test execute_query exception
    res_q = adapter.execute_query("SELECT")
    assert res_q == []
    adapter.conn.execute.side_effect = None

    class CursorMockClose:
        """Docstring for CursorMockClose."""

        def __init__(self):
            """Docstring for __init__."""
            self.description = True

        def execute(self, q, p):
            """Docstring for execute."""

        def fetchall(self):
            """Docstring for fetchall."""
            return [("c_row",)]

        def close(self):
            """Docstring for close."""

    class ConnMockClose:
        """Docstring for ConnMockClose."""

        def cursor(self):
            """Docstring for cursor."""
            return CursorMockClose()

        def close(self):
            """Docstring for close."""

    adapter.conn = ConnMockClose()
    adapter.execute_with_feedback("SELECT")
    adapter.execute_query("SELECT")
