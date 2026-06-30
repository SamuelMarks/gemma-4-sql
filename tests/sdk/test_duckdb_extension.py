"""Tests for DuckDB extension."""

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
    importlib = __import__("importlib")

    mock_duckdb = MagicMock()
    with patch.dict("sys.modules", {"duckdb": mock_duckdb}):
        gemma_4_sql = __import__("gemma_4_sql.sdk.duckdb_extension")
        importlib.reload(gemma_4_sql.sdk.duckdb_extension)
        embed_in_duckdb = __import__("gemma_4_sql.sdk.duckdb_extension", fromlist=["embed_in_duckdb"]).embed_in_duckdb
        conn = MagicMock()

        def mock_execute(query: object) -> object:
            """Initialize function mock_execute.

            Args:
            ----
            query: Description of query.

            """
            mock_cursor = MagicMock()
            if "information_schema.tables" in query:  # type: ignore[operator]
                mock_cursor.fetchall.return_value = [("users",)]
            elif "information_schema.columns" in query:  # type: ignore[operator]
                mock_cursor.fetchall.return_value = [("id", "INTEGER")]
            return mock_cursor

        conn.execute = mock_execute
        registered_func = None

        def mock_create_function(_name: object, func: object, _args: object, _ret: object) -> object:  # type: ignore[return]
            """Initialize function mock_create_function.

            Args:
            ----
            name: Description of name.
            func: Description of func.
            args: Description of args.
            ret: Description of ret.

            """
            nonlocal registered_func
            registered_func = func

        conn.create_function = mock_create_function
        embed_in_duckdb(conn, "model", "jax", ":memory:")
        if not registered_func is not None:
            raise AssertionError
        with patch("gemma_4_sql.sdk.duckdb_extension.run_agentic_loop") as mock_agent:
            mock_agent.return_value = {"final_sql": "SELECT * FROM users", "results": [(1,)], "success": True}
            json = __import__("json")
            res_str = registered_func("Get users")  # type: ignore[operator]
            res_json = json.loads(res_str)
            if res_json["success"] is not True:
                raise AssertionError
            if not res_json["generated_sql"] == "SELECT * FROM users":
                raise AssertionError
            if not res_json["results"] == [[1]]:
                raise AssertionError
            mock_agent.assert_called_once()
            kwargs = mock_agent.call_args.kwargs
            if not kwargs["ddl"] == "CREATE TABLE users (id INTEGER);":
                raise AssertionError
