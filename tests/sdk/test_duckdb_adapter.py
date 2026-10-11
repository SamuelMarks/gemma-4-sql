"""Tests for DuckDB adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemma_4_sql.sdk.adapters.duckdb_adapter import DuckDBAdapter
from tests.conftest import requires_duckdb


@pytest.mark.asyncio
@requires_duckdb
async def test_duckdb_adapter_live(tmp_path: Path) -> None:
    """Test duckdb adapter with real duckdb execution."""
    db_path = str(tmp_path / "test.db")

    # 1. Create and setup schema
    adapter = DuckDBAdapter(db_path, {})
    adapter.setup_schema("CREATE TABLE products (id INTEGER, name TEXT); INSERT INTO products VALUES (1, 'apple');")

    # 2. Query async
    res = await adapter.execute_query_async("SELECT * FROM products;")
    assert res == [(1, "apple")]

    # 3. Query with feedback success
    success, rows, err = await adapter.execute_with_feedback_async("SELECT name FROM products WHERE id=1")
    assert success is True
    assert rows == [("apple",)]
    assert err is None

    # 4. Query with feedback failure
    success, rows, err = await adapter.execute_with_feedback_async("SELECT * FROM missing_table")
    assert success is False
    assert rows == []
    assert "missing_table" in err

    adapter.close()

    # 5. Read-only connection
    read_only_adapter = DuckDBAdapter(db_path, {}, read_only=True)
    res_ro = await read_only_adapter.execute_query_async("SELECT name FROM products")
    assert res_ro == [("apple",)]

    success, _, err = await read_only_adapter.execute_with_feedback_async("INSERT INTO products VALUES (2, 'banana')")
    assert success is False
    assert "read-only" in err.lower() or "read_only" in err.lower() or "not allowed" in err.lower() or "read only" in err.lower() or "cannot" in err.lower()

    read_only_adapter.close()

    # 6. Existing connection
    import duckdb

    conn = duckdb.connect(db_path)
    existing_adapter = DuckDBAdapter(db_path, {"existing_conn": conn})
    res_ex = await existing_adapter.execute_query_async("SELECT name FROM products")
    assert res_ex == [("apple",)]

    # Test with 'conn' instead of 'existing_conn'
    existing_adapter_conn = DuckDBAdapter(db_path, {"conn": conn})
    res_ex_conn = await existing_adapter_conn.execute_query_async("SELECT name FROM products")
    assert res_ex_conn == [("apple",)]

    # Test execute_query_async exception
    res_err = await existing_adapter.execute_query_async("SELECT * FROM non_existent_table")
    assert res_err == []

    # Existing connections should not be closed by adapter
    existing_adapter.close()
    existing_adapter_conn.close()
    # The connection should still be open
    assert conn.execute("SELECT 1").fetchall() == [(1,)]
    conn.close()


def test_duckdb_adapter_missing_duckdb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test DuckDBAdapter raises ImportError when duckdb is missing."""
    import gemma_4_sql.sdk.adapters.duckdb_adapter as dda

    monkeypatch.setattr(dda, "duckdb", None)

    with pytest.raises(ImportError, match="duckdb is required"):
        dda.DuckDBAdapter(":memory:", {}).connect()
