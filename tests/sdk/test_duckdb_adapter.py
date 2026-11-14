"""Unit tests for the DuckDB database adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import duckdb
import pytest

import gemma_4_sql.sdk.adapters.duckdb_adapter as dda
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.sdk.adapters.duckdb_adapter import DuckDBAdapter
from gemma_4_sql.type_hints import JSONPrimitive
from tests.conftest import requires_duckdb


@pytest.mark.asyncio
@requires_duckdb
async def test_duckdb_adapter_live(tmp_path: Path) -> None:
    """Verify `DuckDBAdapter` sync/async queries, read-only `enable_external_access=false`, and `existing_conn`."""
    import duckdb

    db_path = str(tmp_path / "test.db")

    adapter = DuckDBAdapter(db_path, {})
    adapter.setup_schema("CREATE TABLE products (id INTEGER, name TEXT); INSERT INTO products VALUES (1, 'apple');")
    with pytest.raises(DatabaseExecutionError):
        adapter.setup_schema("INVALID DUCKDB DDL")

    res = await adapter.execute_query_async("SELECT * FROM products;")
    assert res == [(1, "apple")]

    conn_async = await adapter.connect_async()
    assert conn_async is adapter.conn

    success, rows, err = await adapter.execute_with_feedback_async("SELECT name FROM products WHERE id=1")
    assert success is True
    assert rows == [("apple",)]
    assert err is None

    success, rows, err = await adapter.execute_with_feedback_async("SELECT * FROM missing_table")
    assert success is False
    assert rows == []
    assert err is not None
    assert "missing_table" in err

    adapter.close()

    read_only_adapter = DuckDBAdapter(db_path, {}, read_only=True)
    res_ro = await read_only_adapter.execute_query_async("SELECT name FROM products")
    assert res_ro == [("apple",)]

    success, _, err = await read_only_adapter.execute_with_feedback_async("INSERT INTO products VALUES (2, 'banana')")
    assert success is False
    assert err is not None
    assert any(tok in err.lower() for tok in ("read-only", "read_only", "not allowed", "read only", "cannot"))

    read_only_adapter.close()

    conn = duckdb.connect(db_path)
    existing_adapter = DuckDBAdapter(db_path, {"existing_conn": conn})
    existing_adapter.setup_schema("")
    existing_adapter.setup_schema("CREATE TABLE extra_tbl (id INT); INSERT INTO extra_tbl VALUES (7);")
    res_extra = await existing_adapter.execute_query_async("SELECT id FROM extra_tbl")
    assert res_extra == [(7,)]
    res_ex = await existing_adapter.execute_query_async("SELECT name FROM products")
    assert res_ex == [("apple",)]

    existing_adapter_conn = DuckDBAdapter(db_path, {"conn": conn})
    res_ex_conn = await existing_adapter_conn.execute_query_async("SELECT name FROM products")
    assert res_ex_conn == [("apple",)]

    with pytest.raises(DatabaseExecutionError, match="non_existent_table"):
        await existing_adapter.execute_query_async("SELECT * FROM non_existent_table")

    existing_adapter.close()
    existing_adapter_conn.close()
    assert conn.execute("SELECT 1").fetchall() == [(1,)]
    conn.close()


def test_duckdb_adapter_missing_and_readonly_pragma_fallback() -> None:
    """Verify `DuckDBAdapter` handles missing `duckdb` and read-only `SET enable_external_access` errors via DI."""
    with pytest.raises(DependencyMissingError, match="duckdb is required"):
        dda.DuckDBAdapter(":memory:", {}, duckdb_module=None)

    class _RestrictedDuckDBConn:
        """Typed DuckDB connection double that rejects `SET enable_external_access`."""

        def execute(
            self,
            query: str,
            _parameters: Sequence[JSONPrimitive] | Mapping[str, JSONPrimitive] | None = None,
            /,
        ) -> _RestrictedDuckDBConn:
            """Raise `RuntimeError` on `SET` and return `self` otherwise.

            Returns:
                `self` when query does not modify `enable_external_access`.

            Raises:
                RuntimeError: When `enable_external_access` appears in `query`.

            """
            if "enable_external_access" in query:
                raise RuntimeError("already locked")
            return self

        def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
            """Return single row."""
            return [(1,)]

        def close(self) -> None:
            """No-op close."""

    class _FakeDuckDBModule:
        """Typed `DuckDBModuleProtocol` double."""

        __name__: str = "duckdb"
        Error: type[Exception] = RuntimeError

        def connect(
            self,
            database: str = ":memory:",
            *,
            read_only: bool = False,
            **_kwargs: str | int | bool,
        ) -> _RestrictedDuckDBConn:
            """Return `_RestrictedDuckDBConn`."""
            _ = (database, read_only)
            return _RestrictedDuckDBConn()

    ro_mem = dda.DuckDBAdapter(":memory:", {}, read_only=True, duckdb_module=_FakeDuckDBModule())
    assert ro_mem.execute_query("SELECT 1") == [(1,)]
    import asyncio

    assert asyncio.run(ro_mem.execute_query_async("SELECT 1")) == [(1,)]

    class _CursorOnlyConn:
        """Cursor-only connection double without direct `.execute`."""

        def __init__(self) -> None:
            """Track DDL statements."""
            self.ddls: list[str] = []

        def cursor(self) -> _RestrictedDuckDBConn:
            """Return `_RestrictedDuckDBConn`."""
            self.ddls.append("cur")
            return _RestrictedDuckDBConn()

    cur_only = _CursorOnlyConn()
    ro_mem.conn = cur_only
    ro_mem.setup_schema("CREATE TABLE t (x INT)")
    assert cur_only.ddls == ["cur"]
    ro_mem.close()

    class _NonTypeErrorDuckDB:
        """DuckDB stub with non-type `Error` attribute."""

        __name__: str = "duckdb"
        Error = 123

    non_type_adapter = dda.DuckDBAdapter(
        ":memory:",
        {},
        lazy_connect=True,
        duckdb_module=_NonTypeErrorDuckDB(),
    )
    assert DatabaseExecutionError in non_type_adapter.error_classes


@requires_duckdb
def test_duckdb_adapter_with_in_memory_fixture(
    in_memory_duckdb: duckdb.DuckDBPyConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `DuckDBAdapter` works with `in_memory_duckdb` and handles `_LoaderSentinel` lazy resolution."""
    monkeypatch.setattr(dda, "duckdb", dda._LoaderSentinel.UNRESOLVED)
    adapter = DuckDBAdapter(":memory:", {"existing_conn": in_memory_duckdb})
    assert adapter.execute_query("SELECT 42") == [(42,)]
    assert len(adapter.error_classes) == 1
    missing_mod_adapter = DuckDBAdapter(":memory:", {"existing_conn": in_memory_duckdb}, duckdb_module=None)
    assert DatabaseExecutionError in missing_mod_adapter.error_classes
    adapter.close()
