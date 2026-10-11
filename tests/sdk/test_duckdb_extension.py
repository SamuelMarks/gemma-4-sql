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
        res_str = conn.execute("SELECT ask_gemma('Get users')").fetchone()[0]

        import json

        res_json = json.loads(res_str)
        assert res_json["success"] is True
        assert res_json["generated_sql"] == "SELECT * FROM users"
        assert res_json["results"] == [[42]]

        mock_agent.assert_called_once()
        kwargs = mock_agent.call_args.kwargs
        assert "CREATE TABLE users" in kwargs["context"].ddl


def test_embed_in_duckdb_missing(missing_dependency_context: Any) -> None:  # noqa: F821
    """Test when duckdb is missing."""
    with missing_dependency_context(["duckdb"]):
        import importlib

        import gemma_4_sql.sdk.duckdb_extension as de

        importlib.reload(de)
        with pytest.raises(ImportError, match="duckdb is required"):
            de.embed_in_duckdb(MagicMock(), "model", "jax")
