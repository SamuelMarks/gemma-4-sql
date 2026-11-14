"""Unit tests for `SnowflakeAdapter` using strongly-typed protocol test doubles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self

import pytest

import gemma_4_sql.sdk.adapters.snowflake_adapter as sa
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.sdk.adapters.base import DBKwargValue, MinimalCursorProtocol
from gemma_4_sql.type_hints import DatabaseConnectionProtocol, JSONPrimitive
from tests.utils.test_utils import FakeDBConnection


class _SnowflakeError(Exception):
    """Simulated Snowflake connector error."""


class _SnowflakeErrorsNS:
    """Typed `snowflake.connector.errors` namespace."""

    def __init__(self, err_val: type[Exception] | int = _SnowflakeError) -> None:
        """Store `Error` attribute.

        Args:
            err_val: Exception class or invalid sentinel value.

        """
        self.Error = err_val


class _SnowflakeConnectorSubmod:
    """Typed `snowflake.connector` submodule double."""

    def __init__(self, conn: FakeDBConnection) -> None:
        """Initialize with `FakeDBConnection` and call counter.

        Args:
            conn: Underlying fake connection.

        """
        self._conn = conn
        self.connect_calls = 0
        self.errors = _SnowflakeErrorsNS(_SnowflakeError)

    def connect(self, **kwargs: DBKwargValue) -> DatabaseConnectionProtocol:
        """Increment `connect_calls` and return `self._conn`.

        Args:
            **kwargs: Connection parameters.

        Returns:
            `FakeDBConnection` instance.

        """
        _ = kwargs
        self.connect_calls += 1
        return self._conn


class _SnowflakeRootPkg:
    """Typed top-level `snowflake` package double."""

    def __init__(self, conn: FakeDBConnection) -> None:
        """Initialize with `_SnowflakeConnectorSubmod`.

        Args:
            conn: Underlying fake connection.

        """
        self.connector = _SnowflakeConnectorSubmod(conn)


class _DirectSnowflakeModule:
    """Typed direct `snowflake.connector` module exposing `.connect(...)`."""

    def __init__(self, conn: FakeDBConnection) -> None:
        """Store `FakeDBConnection`.

        Args:
            conn: Underlying fake connection.

        """
        self._conn = conn

    def connect(self, **kwargs: DBKwargValue) -> DatabaseConnectionProtocol:
        """Return `self._conn`.

        Args:
            **kwargs: Connection parameters.

        Returns:
            `FakeDBConnection` instance.

        """
        _ = kwargs
        return self._conn


class _EmptySnowflakeObject:
    """Module stub with neither `.connector` nor `.connect`."""

    __name__: str = "snowflake"


@pytest.fixture
def fake_snowflake_bundle() -> tuple[_SnowflakeRootPkg, FakeDBConnection]:
    """Provide typed `snowflake` package double and `FakeDBConnection`.

    Returns:
        Tuple of `(snowflake_pkg, fake_conn)`.

    """
    conn = FakeDBConnection()
    return (_SnowflakeRootPkg(conn), conn)


@pytest.mark.asyncio
async def test_snowflake_adapter_execution_and_no_async_leak(
    fake_snowflake_bundle: tuple[_SnowflakeRootPkg, FakeDBConnection],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `SnowflakeAdapter` execution, error propagation, and leak-free `connect_async`."""
    sf_pkg, conn = fake_snowflake_bundle

    adapter = sa.SnowflakeAdapter("account", {}, snowflake_module=sf_pkg)
    assert sf_pkg.connector.connect_calls == 1
    # `connect_async()` reuses the already-established `self._conn` without leaking a duplicate connection!
    c_async = await adapter.connect_async()
    assert c_async is conn
    assert sf_pkg.connector.connect_calls == 1

    # Lazy adapter connects on first `connect_async()` call
    lazy_sf = sa.SnowflakeAdapter("account", {}, lazy_connect=True, snowflake_module=sf_pkg)
    assert sf_pkg.connector.connect_calls == 1
    assert await lazy_sf.connect_async() is conn
    assert sf_pkg.connector.connect_calls == 2

    assert _SnowflakeError in adapter.error_classes
    sf_pkg.connector.errors = _SnowflakeErrorsNS(123)
    assert DatabaseExecutionError in adapter.error_classes
    sf_pkg.connector.errors = _SnowflakeErrorsNS(_SnowflakeError)

    # Direct `SnowflakeModuleProtocol` fallback (`sf_mod.connect`) and unresolvable module error
    class _NoErrorsRootPkg:
        """Package stub with `.connector` lacking `.errors`."""

        __name__: str = "snowflake"
        connector = _DirectSnowflakeModule(conn)

    no_err_adapter = sa.SnowflakeAdapter("account", {}, lazy_connect=True, snowflake_module=_NoErrorsRootPkg())
    assert DatabaseExecutionError in no_err_adapter.error_classes
    direct_adapter = sa.SnowflakeAdapter(
        "account", {}, lazy_connect=True, snowflake_module=_DirectSnowflakeModule(conn)
    )
    assert direct_adapter.connect() is conn
    empty_sf_adapter = sa.SnowflakeAdapter("account", {}, lazy_connect=True, snowflake_module=_EmptySnowflakeObject())
    with pytest.raises(DependencyMissingError, match="snowflake connect function could not be resolved"):
        empty_sf_adapter.connect()

    conn.set_results([[(1, "apple")]])
    assert await adapter.execute_query_async("SELECT") == [(1, "apple")]

    conn.set_results([[(1, "apple")]])
    suc, rows, err = await adapter.execute_with_feedback_async("SELECT")
    assert suc is True and rows == [(1, "apple")] and err is None

    class _FailingCursor:
        """Cursor double raising `_SnowflakeError`."""

        description: Sequence[Sequence[str | int | None]] | bool | None = None

        def execute(
            self,
            _q: str,
            _p: Sequence[JSONPrimitive] | Mapping[str, JSONPrimitive] | None = None,
            /,
        ) -> Self:
            """Raise `_SnowflakeError`.

            Raises:
                _SnowflakeError: Always raised to simulate Snowflake execution failure.

            """
            raise _SnowflakeError("Simulated")

        def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
            """Return empty rows."""
            return []

        def close(self) -> None:
            """No-op close."""

    assert _FailingCursor().fetchall() == []
    monkeypatch.setattr(conn, "cursor", _FailingCursor)
    suc_f, _, err_f = await adapter.execute_with_feedback_async("FAIL")
    assert suc_f is False and "Simulated" in str(err_f)
    with pytest.raises(DatabaseExecutionError, match="Simulated"):
        await adapter.execute_query_async("FAIL")


def test_snowflake_adapter_missing_deps_and_lazy_loader(
    fake_snowflake_bundle: tuple[_SnowflakeRootPkg, FakeDBConnection],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `SnowflakeAdapter` raises `DependencyMissingError` when Snowflake is uninstalled."""
    monkeypatch.setattr(sa, "snowflake", sa._LoaderSentinel.UNRESOLVED)
    _ = sa._resolve_snowflake()

    sf_pkg, _ = fake_snowflake_bundle
    monkeypatch.setattr(sa, "snowflake", sf_pkg)
    adapter = sa.SnowflakeAdapter("path", {})

    monkeypatch.setattr(sa, "snowflake", None)
    with pytest.raises(DependencyMissingError, match="snowflake-connector-python is required"):
        sa.SnowflakeAdapter("path", {})
    assert DatabaseExecutionError in adapter.error_classes


def test_snowflake_adapter_schema(
    fake_snowflake_bundle: tuple[_SnowflakeRootPkg, FakeDBConnection],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `SnowflakeAdapter.setup_schema` commits on success and wraps errors in `DatabaseExecutionError`."""
    sf_pkg, conn = fake_snowflake_bundle
    monkeypatch.setattr(sa, "snowflake", sf_pkg)

    adapter = sa.SnowflakeAdapter("path", {})
    adapter.setup_schema("CREATE TABLE t (a INT)")
    assert conn._transactions_committed == 1

    conn.set_fail_query("FAIL")
    with pytest.raises(DatabaseExecutionError, match="Simulated execution failure"):
        adapter.setup_schema("FAIL")
    assert conn._transactions_rolled_back == 1

    class StatelessCursorConn:
        """Connection stub without commit/rollback methods."""

        def __init__(self, *, should_fail: bool = False) -> None:
            """Initialize with optional failure flag."""
            self.should_fail = should_fail

        def execute(
            self,
            _q: str,
            _p: Sequence[JSONPrimitive] | Mapping[str, JSONPrimitive] | None = None,
            /,
        ) -> MinimalCursorProtocol:
            """Delegate to cursor.

            Returns:
                Minimal cursor instance.

            """
            return self.cursor()

        def cursor(self) -> MinimalCursorProtocol:
            """Return a minimal cursor without `.close()`."""
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
                    """Execute DDL or raise `RuntimeError`.

                    Returns:
                        `self`.

                    Raises:
                        RuntimeError: When `outer.should_fail` is enabled.

                    """
                    if outer.should_fail:
                        raise RuntimeError("Mock error")
                    return self

                def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
                    """Return empty result set."""
                    return []

            return _Cur()

        def close(self) -> None:
            """Close connection no-op."""

    sf_ok = StatelessCursorConn(should_fail=False)
    assert sf_ok.execute("SELECT 1").fetchall() == []
    adapter.conn = sf_ok
    adapter.setup_schema("CREATE TABLE t2 (a INT)")

    adapter.conn = StatelessCursorConn(should_fail=True)
    with pytest.raises(DatabaseExecutionError, match="Mock error"):
        adapter.setup_schema("FAIL2")
