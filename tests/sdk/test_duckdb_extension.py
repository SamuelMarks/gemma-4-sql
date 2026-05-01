"""
Tests for DuckDB extension.
"""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.sdk.duckdb_extension import embed_in_duckdb


@patch("gemma_4_sql.sdk.duckdb_extension.duckdb", new=None)
def test_embed_in_duckdb_missing() -> None:
    """Test when duckdb is missing."""
    with pytest.raises(ImportError, match="duckdb is required"):
        embed_in_duckdb(MagicMock(), "model", "jax")


def test_embed_in_duckdb_success() -> None:
    """Test successful registration and execution."""
    import importlib
    from unittest.mock import MagicMock, patch

    mock_duckdb = MagicMock()

    with patch.dict("sys.modules", {"duckdb": mock_duckdb}):
        import gemma_4_sql.sdk.duckdb_extension

        importlib.reload(gemma_4_sql.sdk.duckdb_extension)
        from gemma_4_sql.sdk.duckdb_extension import embed_in_duckdb

        conn = MagicMock()

        # Mock schema extraction
        def mock_execute(query):
            mock_cursor = MagicMock()
            if "information_schema.tables" in query:
                mock_cursor.fetchall.return_value = [("users",)]
            elif "information_schema.columns" in query:
                mock_cursor.fetchall.return_value = [("id", "INTEGER")]
            return mock_cursor

        conn.execute = mock_execute

        # We need to capture the registered function to test it
        registered_func = None

        def mock_create_function(name, func, args, ret):
            nonlocal registered_func
            registered_func = func

        conn.create_function = mock_create_function

        embed_in_duckdb(conn, "model", "jax", ":memory:")

        assert registered_func is not None

        # Now mock run_agentic_loop
        with patch("gemma_4_sql.sdk.duckdb_extension.run_agentic_loop") as mock_agent:
            mock_agent.return_value = {
                "final_sql": "SELECT * FROM users",
                "results": [(1,)],
                "success": True,
            }
            import json

            res_str = registered_func("Get users")
            res_json = json.loads(res_str)

            assert res_json["success"] is True
            assert res_json["generated_sql"] == "SELECT * FROM users"
            assert res_json["results"] == [[1]]  # JSON converts tuple to list

            # Verify DDL passed to agent
            mock_agent.assert_called_once()
            kwargs = mock_agent.call_args.kwargs
            assert kwargs["ddl"] == "CREATE TABLE users (id INTEGER);"
