"""Unit tests for the Snowflake database adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.sdk.adapters.snowflake_adapter as sa
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.type_hints import CursorProtocol, JSONPrimitive, JSONValue
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
        assert DatabaseExecutionError in adapter.error_classes

        original_connector = mock_snowflake.connector
        del mock_snowflake.connector
        mock_snowflake.connect = lambda **_kw: conn  # pyright: ignore[reportUnknownLambdaType]
        assert adapter.connect() == conn
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

        def mock_fail(*_a: str | Sequence[JSONPrimitive], **_k: JSONValue) -> None:
            """Raise simulated SnowflakeError."""
            msg = "Simulated"
            raise snowflake_error_cls(msg)

        mock_cur_fn = MagicMock()
        mock_cur_fn.return_value.execute = mock_fail
        monkeypatch.setattr(conn, "cursor", mock_cur_fn)
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

    assert DatabaseExecutionError in adapter.error_classes


def test_snowflake_adapter_schema(
    fake_snowflake_module: tuple[MagicMock, FakeDBConnection, type[Exception]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SnowflakeAdapter.setup_schema commits on success and wraps errors in DatabaseExecutionError."""
    mock_snowflake, conn, _ = fake_snowflake_module

    monkeypatch.setattr(sa, "snowflake", mock_snowflake)

    adapter = sa.SnowflakeAdapter("path", {})
    adapter.setup_schema("CREATE TABLE t (a INT)")
    assert conn._transactions_committed == 1

    conn.set_fail_query("FAIL")
    with pytest.raises(DatabaseExecutionError, match="Simulated execution failure"):
        adapter.setup_schema("FAIL")
    assert conn._transactions_rolled_back == 1

    class StatelessCursorConn:
        """Connection stub without commit/rollback methods."""

        def __init__(self, should_fail: bool = False) -> None:
            """Initialize with optional failure flag."""
            self.should_fail = should_fail

        def cursor(self) -> CursorProtocol:
            """Return a minimal cursor."""
            outer = self

            class _Cur:
                """Minimal cursor stub for DDL execution."""

                description: Sequence[Sequence[str | int | None]] | bool | None = None

                def execute(
                    self,
                    _q: str,
                    _params: Sequence[JSONPrimitive] | Mapping[str, JSONPrimitive] | None = None,
                    /,
                ) -> Self:
                    """Execute DDL or raise RuntimeError."""
                    if outer.should_fail:
                        raise RuntimeError("Mock error")
                    return self

                def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
                    """Return empty result set."""
                    return []

                def close(self) -> None:
                    """Close cursor no-op."""

            return _Cur()

        def close(self) -> None:
            """Close connection no-op."""

    sf_ok = StatelessCursorConn(should_fail=False)
    assert sf_ok.cursor().fetchall() == []
    monkeypatch.setattr(adapter, "conn", sf_ok)
    adapter.setup_schema("CREATE TABLE t2 (a INT)")

    monkeypatch.setattr(adapter, "conn", StatelessCursorConn(should_fail=True))
    with pytest.raises(DatabaseExecutionError, match="Mock error"):
        adapter.setup_schema("FAIL2")
