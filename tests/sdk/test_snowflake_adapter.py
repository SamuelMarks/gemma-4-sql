"""Module docstring."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_snowflake_adapter(monkeypatch):
    """Docstring for test_snowflake_adapter."""
    import importlib

    import gemma_4_sql.sdk.adapters.snowflake_adapter as sa

    # error_classes with missing snowflake module
    with patch.dict("sys.modules", {"snowflake": None, "snowflake.connector": None}):
        importlib.reload(sa)
        adapter = sa.SnowflakeAdapter.__new__(sa.SnowflakeAdapter)
        assert adapter.error_classes == (Exception,)

    # error_classes with proper exception class
    class DummyError(Exception):
        """Docstring for DummyError."""

    mock_snowflake = MagicMock()
    mock_snowflake.connector = type("connector", (), {"errors": type("errors", (), {"Error": DummyError})()})()
    with patch.dict("sys.modules", {"snowflake": mock_snowflake, "snowflake.connector": mock_snowflake.connector}):
        importlib.reload(sa)
        adapter = sa.SnowflakeAdapter.__new__(sa.SnowflakeAdapter)
        # Instead of calling it via the class we check if we patched it correctly, but error_classes returns the class, so we can just assert on the property directly now because reload will set sa.snowflake to our mock!
        # Wait, the property re-imports it? Let's see:
        assert adapter.error_classes == (DummyError,)

    # error_classes with snowflake module present but missing Exception base
    mock_snowflake_bad = MagicMock()
    mock_snowflake_bad.connector = type("connector", (), {"errors": type("errors", (), {"Error": int})()})()
    with patch.dict("sys.modules", {"snowflake": mock_snowflake_bad, "snowflake.connector": mock_snowflake_bad.connector}):
        importlib.reload(sa)
        adapter = sa.SnowflakeAdapter.__new__(sa.SnowflakeAdapter)
        assert adapter.error_classes == (Exception,)

    # error_classes with AttributeError
    mock_snowflake_attr = MagicMock()
    del mock_snowflake_attr.connector
    with patch.dict("sys.modules", {"snowflake": mock_snowflake_attr}):
        importlib.reload(sa)
        adapter = sa.SnowflakeAdapter.__new__(sa.SnowflakeAdapter)
        assert adapter.error_classes == (Exception,)

    # restore module state so sa is normal
    importlib.reload(sa)

    # connect sync
    adapter = sa.SnowflakeAdapter.__new__(sa.SnowflakeAdapter)
    adapter.db_kwargs = {}
    with patch.object(sa, "snowflake", mock_snowflake):
        mock_snowflake.connector.connect = MagicMock(return_value="conn")
        assert adapter.connect() == "conn"

    # connect sync missing connect
    mock_snowflake_noconnect = MagicMock()
    del mock_snowflake_noconnect.connector
    del mock_snowflake_noconnect.connect
    with patch.object(sa, "snowflake", mock_snowflake_noconnect):
        with pytest.raises(ImportError):
            adapter.connect()

    # connect sync missing module
    with patch.object(sa, "snowflake", None):
        with pytest.raises(ImportError):
            adapter.connect()

    # connect async
    adapter.connect = MagicMock(return_value="aconn")
    res = await adapter.connect_async()
    assert res == "aconn"

    # setup_schema success
    adapter.conn = MagicMock()
    adapter.setup_schema("DDL")
    adapter.conn.cursor.return_value.execute.assert_called_with("DDL")
    adapter.conn.commit.assert_called_once()

    # setup_schema success NO commit
    del adapter.conn.commit
    adapter.setup_schema("DDL")

    # setup_schema error
    adapter.conn = MagicMock()
    adapter.conn.cursor.return_value.execute.side_effect = ValueError("err")
    with pytest.raises(ValueError):
        adapter.setup_schema("DDL")
    adapter.conn.rollback.assert_called_once()

    # setup_schema error NO rollback
    adapter.conn = MagicMock()
    del adapter.conn.rollback
    adapter.conn.cursor.return_value.execute.side_effect = ValueError("err")
    with pytest.raises(ValueError):
        adapter.setup_schema("DDL")

    # async wrappers
    adapter.execute_with_feedback = MagicMock(return_value="fw_res")
    assert await adapter.execute_with_feedback_async("SELECT") == "fw_res"

    adapter.execute_query = MagicMock(return_value="q_res")
    assert await adapter.execute_query_async("SELECT") == "q_res"
