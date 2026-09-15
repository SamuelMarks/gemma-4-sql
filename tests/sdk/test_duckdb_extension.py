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
    """Test successful registration and execution.

    Raises:
        AssertionError: Description.

    """
    importlib = __import__("importlib", fromlist=[""])
    mock_duckdb = MagicMock()
    with patch.dict("sys.modules", {"duckdb": mock_duckdb}):
        gemma_4_sql = __import__("gemma_4_sql.sdk.duckdb_extension")
        importlib.reload(gemma_4_sql.sdk.duckdb_extension)
        embed_in_duckdb = __import__("gemma_4_sql.sdk.duckdb_extension", fromlist=["embed_in_duckdb"]).embed_in_duckdb
        conn = MagicMock()

        def mock_execute(query: object, *args: object, **kwargs: object) -> object:
            """Initialize function mock_execute.

            Args:
            ----
            query: Description of query.
            *args: Optional arguments.
            **kwargs: Optional keyword arguments.

            Returns:
                object: Description of return.

            """
            mock_cursor = MagicMock()
            if "information_schema.tables" in query:
                mock_cursor.fetchall.return_value = [("users",)]
            elif "information_schema.columns" in query:
                mock_cursor.fetchall.return_value = [("id", "INTEGER")]
            return mock_cursor

        conn.execute = mock_execute
        registered_func = None

        def mock_create_function(_name: object, func: object, _args: object, _ret: object) -> object:
            """Initialize function mock_create_function.

            Args:
            ----
            func: Description of func.

            """
            nonlocal registered_func
            registered_func = func

        conn.create_function = mock_create_function
        embed_in_duckdb(conn, "model", "jax", ":memory:")
        if not registered_func is not None:
            raise AssertionError
        with patch("gemma_4_sql.sdk.duckdb_extension.run_agentic_loop") as mock_agent:
            mock_agent.return_value = {"final_sql": "SELECT * FROM users", "results": [(1,)], "success": True}
            json = __import__("json", fromlist=[""])
            res_str = registered_func("Get users")
            res_json = json.loads(res_str)
            if res_json["success"] is not True:
                raise AssertionError
            if not res_json["generated_sql"] == "SELECT * FROM users":
                raise AssertionError
            if not res_json["results"] == [[1]]:
                raise AssertionError
            mock_agent.assert_called_once()
            kwargs = mock_agent.call_args.kwargs
            if not kwargs["context"].ddl == "CREATE TABLE users (id INTEGER);":
                raise AssertionError


def test_duckdb_adapter_missing_duckdb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test DuckDBAdapter raises ImportError when duckdb is missing."""
    import gemma_4_sql.sdk.adapters.duckdb_adapter as dda

    monkeypatch.setattr(dda, "duckdb", None)
    with pytest.raises(ImportError, match="duckdb is required"):
        dda.DuckDBAdapter(":memory:", {})


def test_duckdb_adapter_external_conn_close() -> None:
    """Test DuckDBAdapter does not close an externally provided connection."""
    import gemma_4_sql.sdk.adapters.duckdb_adapter as dda

    mock_conn = MagicMock()
    ad = dda.DuckDBAdapter(":memory:", {"existing_conn": mock_conn})
    ad.setup_schema("CREATE TABLE t (x INT);")
    ad.close()
    mock_conn.close.assert_not_called()

    ad2 = dda.DuckDBAdapter(":memory:", {"conn": mock_conn})
    ad2.setup_schema("CREATE TABLE t (x INT);")
    ad2.close()
    mock_conn.close.assert_not_called()


def test_duckdb_adapter_readonly_file(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test DuckDBAdapter with read_only flag on disk path."""
    import gemma_4_sql.sdk.adapters.duckdb_adapter as dda

    mock_duckdb = MagicMock()
    monkeypatch.setattr(dda, "duckdb", mock_duckdb)
    ad = dda.DuckDBAdapter(str(tmp_path) + "/test.db", {}, read_only=True)
    ad.close()
    mock_duckdb.connect.assert_called_once_with(str(tmp_path) + "/test.db", read_only=True)
