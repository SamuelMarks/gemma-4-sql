"""Unit tests for the DuckDB database adapter."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

import gemma_4_sql.sdk.adapters.duckdb_adapter as dda
from gemma_4_sql.exceptions import DatabaseExecutionError, DependencyMissingError
from gemma_4_sql.sdk.adapters.duckdb_adapter import DuckDBAdapter
from tests.conftest import requires_duckdb


@pytest.mark.asyncio
@requires_duckdb
async def test_duckdb_adapter_live(tmp_path: Path) -> None:
    """Verify DuckDBAdapter synchronous/asynchronous queries, read-only mode, and existing_conn handling."""
    import duckdb

    db_path = str(tmp_path / "test.db")

    adapter = DuckDBAdapter(db_path, {})
    adapter.setup_schema("CREATE TABLE products (id INTEGER, name TEXT); INSERT INTO products VALUES (1, 'apple');")

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


def test_duckdb_adapter_missing_duckdb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DuckDBAdapter raises DependencyMissingError when duckdb is missing."""
    monkeypatch.setattr(dda, "duckdb", None)

    with pytest.raises(DependencyMissingError, match="duckdb is required"):
        dda.DuckDBAdapter(":memory:", {})


@requires_duckdb
def test_duckdb_adapter_with_in_memory_fixture(
    in_memory_duckdb: duckdb.DuckDBPyConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify DuckDBAdapter works with the shared in_memory_duckdb fixture and handles missing duckdb.Error."""
    adapter = DuckDBAdapter(":memory:", {"existing_conn": in_memory_duckdb})
    assert adapter.execute_query("SELECT 42") == [(42,)]
    monkeypatch.setattr(dda, "duckdb", None)
    assert DatabaseExecutionError in adapter.error_classes
    adapter.close()
