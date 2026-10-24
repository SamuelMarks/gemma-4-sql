"""Unit tests for the Snowflake database adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.sdk.adapters.snowflake_adapter as sa
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from tests.utils.test_utils import FakeDBConnection


@pytest.fixture
def fake_snowflake_module() -> tuple[MagicMock, FakeDBConnection, type[Exception]]:
    """Create a mocked snowflake.connector module backed by FakeDBConnection.

    Returns:
        Tuple of `(mock_snowflake, fake_conn, SnowflakeError)`.

    """
    mock_snowflake = MagicMock()

    class SnowflakeError(Exception):
        """Simulated Snowflake connector error."""

    mock_snowflake.connector.errors.Error = SnowflakeError

    conn = FakeDBConnection()
    mock_snowflake.connector.connect.return_value = conn
    return mock_snowflake, conn, SnowflakeError


@pytest.mark.asyncio
async def test_snowflake_adapter_execution(
    fake_snowflake_module: tuple[MagicMock, FakeDBConnection, type[Exception]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SnowflakeAdapter connection, query execution, and error propagation."""
    mock_snowflake, conn, snowflake_error_cls = fake_snowflake_module

    monkeypatch.setattr(sa, "snowflake", mock_snowflake)

    with patch.dict("sys.modules", {"snowflake": mock_snowflake, "snowflake.connector": mock_snowflake.connector}):
        adapter = sa.SnowflakeAdapter("account", {})

        c1 = adapter.connect()
        assert c1 == conn

        c2 = await adapter.connect_async()
        assert c2 == conn

        assert snowflake_error_cls in adapter.error_classes

        mock_snowflake.connector.errors.Error = 123
        assert Exception in adapter.error_classes

        original_connector = mock_snowflake.connector
        del mock_snowflake.connector
        mock_snowflake.connect = None
        with pytest.raises(DependencyMissingError, match="snowflake connect function could not be resolved"):
            adapter.connect()
        mock_snowflake.connector = original_connector

        conn.set_results([[(1, "apple")]])
        res = await adapter.execute_query_async("SELECT")
        assert res == [(1, "apple")]

        conn.set_results([[(1, "apple")]])
        suc, res, err = await adapter.execute_with_feedback_async("SELECT")
        assert suc is True
        assert res == [(1, "apple")]
        assert err is None

        mock_snowflake.connector.errors.Error = snowflake_error_cls

        def mock_fail(*_a: Any, **_k: Any) -> None:
            """Raise simulated SnowflakeError."""
            msg = "Simulated"
            raise snowflake_error_cls(msg)

        mock_cur_fn = MagicMock()
        mock_cur_fn.return_value.execute = mock_fail
        setattr(conn, "cursor", mock_cur_fn)
        suc, res, err = await adapter.execute_with_feedback_async("FAIL")
        assert suc is False
        assert "Simulated" in str(err)

        with pytest.raises(DatabaseExecutionError, match="Simulated"):
            await adapter.execute_query_async("FAIL")


@pytest.mark.asyncio
async def test_snowflake_adapter_missing_deps(
    fake_snowflake_module: tuple[MagicMock, FakeDBConnection, type[Exception]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SnowflakeAdapter raises DependencyMissingError when snowflake-connector-python is missing."""
    mock_snowflake, _, _ = fake_snowflake_module
    monkeypatch.setattr(sa, "snowflake", mock_snowflake)
    adapter = sa.SnowflakeAdapter("path", {})

    monkeypatch.setattr(sa, "snowflake", None)
    with pytest.raises(DependencyMissingError, match="snowflake-connector-python is required"):
        sa.SnowflakeAdapter("path", {})

    with patch.dict("sys.modules", {"snowflake.connector": None}):
        assert Exception in adapter.error_classes


def test_snowflake_adapter_schema(
    fake_snowflake_module: tuple[MagicMock, FakeDBConnection, type[Exception]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SnowflakeAdapter.setup_schema commits on success and rolls back on failure."""
    mock_snowflake, conn, _ = fake_snowflake_module

    monkeypatch.setattr(sa, "snowflake", mock_snowflake)

    adapter = sa.SnowflakeAdapter("path", {})
    adapter.setup_schema("CREATE TABLE t (a INT)")
    assert conn._transactions_committed == 1

    conn.set_fail_query("FAIL")
    with pytest.raises(Exception, match="Simulated execution failure"):
        adapter.setup_schema("FAIL")
    assert conn._transactions_rolled_back == 1

    adapter.conn = MagicMock()
    del adapter.conn.commit
    del adapter.conn.rollback

    adapter.setup_schema("CREATE TABLE t2 (a INT)")

    adapter.conn.cursor.return_value.execute.side_effect = RuntimeError("Mock error")
    with pytest.raises(RuntimeError, match="Mock error"):
        adapter.setup_schema("FAIL2")
