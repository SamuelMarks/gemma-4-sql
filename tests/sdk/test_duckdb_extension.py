"""Tests for DuckDB extension."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.sdk.duckdb_extension import embed_in_duckdb
from tests.conftest import requires_duckdb


@requires_duckdb
def test_embed_in_duckdb_success() -> None:
    """Test successful registration and execution."""
    import duckdb

    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER); INSERT INTO users VALUES (42);")

    embed_in_duckdb(conn, "model", "pytorch")

    with patch("gemma_4_sql.sdk.duckdb_extension.run_agentic_loop") as mock_agent:
        mock_agent.return_value = {"final_sql": "SELECT * FROM users", "results": [(42,)], "success": True}

        # Test using the registered function
        row1 = conn.execute("SELECT ask_gemma('Get users')").fetchone()
        assert row1 is not None
        res_str = row1[0]

        import json

        res_json = json.loads(res_str)
        assert res_json["success"] is True
        assert res_json["generated_sql"] == "SELECT * FROM users"
        assert res_json["results"] == [[42]]

        mock_agent.assert_called_once()
        kwargs = mock_agent.call_args.kwargs
        assert "Table: users" in kwargs["prompt"]

        # Second invocation reuses cached DDL without re-extracting
        row2 = conn.execute("SELECT ask_gemma('Get users again')").fetchone()
        assert row2 is not None
        res_str2 = row2[0]
        assert json.loads(res_str2)["success"] is True


def test_embed_in_duckdb_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `embed_in_duckdb` raises `DependencyMissingError` (an `ImportError` subclass) when `duckdb` is None."""
    import gemma_4_sql.sdk.duckdb_extension as de
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(de, "duckdb", None)
    with pytest.raises(DependencyMissingError, match="duckdb is required"):
        de.embed_in_duckdb(MagicMock(), "model", "jax")

    mock_cur = MagicMock()

    def _exec_side_effect(query: str, _params: list[str] | None = None) -> MagicMock:
        """Simulate duckdb_tables() raising RuntimeError and information_schema succeeding."""
        res_mock = MagicMock()
        if "duckdb_tables()" in query:
            raise RuntimeError("duckdb_tables not supported")
        if "information_schema.tables" in query:
            res_mock.fetchall.return_value = [("users",)]
        else:
            assert "table_schema = 'main'" in query
            res_mock.fetchall.return_value = [("id", "INTEGER"), ("name", "VARCHAR")]
        return res_mock

    mock_cur.execute.side_effect = _exec_side_effect
    assert de._extract_duckdb_ddl(mock_cur) == "CREATE TABLE users (id INTEGER, name VARCHAR);"

    mock_cur_empty = MagicMock()
    mock_cur_empty.execute.return_value.fetchall.return_value = []
    assert de._extract_duckdb_ddl(mock_cur_empty) == ""
