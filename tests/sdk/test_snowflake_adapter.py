"""Tests for Snowflake adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.utils.test_utils import FakeDBConnection


@pytest.fixture
def fake_snowflake_module():
    mock_snowflake = MagicMock()

    class SnowflakeError(Exception):
        pass

    mock_snowflake.connector.errors.Error = SnowflakeError

    conn = FakeDBConnection()
    mock_snowflake.connector.connect.return_value = conn
    return mock_snowflake, conn, SnowflakeError


@pytest.mark.asyncio
async def test_snowflake_adapter_execution(fake_snowflake_module, monkeypatch) -> None:
    """Test SnowflakeAdapter execution paths."""
    mock_snowflake, conn, SnowflakeError = fake_snowflake_module
    import gemma_4_sql.sdk.adapters.snowflake_adapter as sa

    monkeypatch.setattr(sa, "snowflake", mock_snowflake)

    with patch.dict("sys.modules", {"snowflake": mock_snowflake, "snowflake.connector": mock_snowflake.connector}):
        adapter = sa.SnowflakeAdapter("account", {})

        # test connect and connect_async
        c1 = adapter.connect()
        assert c1 == conn

        c2 = await adapter.connect_async()
        assert c2 == conn

        # test error classes
        assert SnowflakeError in adapter.error_classes

        # test error classes exception branch
        mock_snowflake.connector.errors.Error = 123
        assert Exception in adapter.error_classes

        # test connect exception
        original_connector = mock_snowflake.connector
        del mock_snowflake.connector
        mock_snowflake.connect = None
        with pytest.raises(ImportError, match="snowflake connect function could not be resolved"):
            adapter.connect()
        mock_snowflake.connector = original_connector

        # execute_query_async delegates to sync
        conn.set_results([[(1, "apple")]])
        res = await adapter.execute_query_async("SELECT")
        assert res == [(1, "apple")]

        conn.set_results([[(1, "apple")]])
        suc, res, err = await adapter.execute_with_feedback_async("SELECT")
        assert suc is True
        assert res == [(1, "apple")]

        # test failure
        def mock_fail(*a, **k):
            raise SnowflakeError("Simulated")

        conn.cursor = MagicMock()
        conn.cursor.return_value.execute = mock_fail
        suc, res, err = await adapter.execute_with_feedback_async("FAIL")
        assert suc is False
        assert "Simulated" in str(err)


@pytest.mark.asyncio
async def test_snowflake_adapter_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test missing dependencies."""
    import gemma_4_sql.sdk.adapters.snowflake_adapter as sa

    monkeypatch.setattr(sa, "snowflake", None)
    with pytest.raises(ImportError):
        sa.SnowflakeAdapter("path", {})

    # Test error_classes import error branch
    with patch.dict("sys.modules", {"snowflake.connector": None}):
        # Mock __init__ to bypass connection init
        with patch.object(sa.SnowflakeAdapter, "__init__", return_value=None):
            dummy_adapter = sa.SnowflakeAdapter("path", {})
            assert Exception in dummy_adapter.error_classes


def test_snowflake_adapter_schema(fake_snowflake_module, monkeypatch) -> None:
    """Test setup_schema."""
    mock_snowflake, conn, SnowflakeError = fake_snowflake_module
    import gemma_4_sql.sdk.adapters.snowflake_adapter as sa

    monkeypatch.setattr(sa, "snowflake", mock_snowflake)

    adapter = sa.SnowflakeAdapter("path", {})
    adapter.setup_schema("CREATE TABLE t (a INT)")
    assert conn._transactions_committed == 1

    conn.set_fail_query("FAIL")
    with pytest.raises(Exception):
        adapter.setup_schema("FAIL")
    assert conn._transactions_rolled_back == 1

    # Test missing commit/rollback attributes
    adapter.conn = MagicMock()
    del adapter.conn.commit
    del adapter.conn.rollback

    adapter.setup_schema("CREATE TABLE t2 (a INT)")

    adapter.conn.cursor.return_value.execute.side_effect = Exception("Mock error")
    with pytest.raises(Exception, match="Mock error"):
        adapter.setup_schema("FAIL2")
