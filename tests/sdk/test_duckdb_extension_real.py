"""Provide module docstring."""

import json

import duckdb
import pytest

from gemma_4_sql.sdk.duckdb_extension import embed_in_duckdb


def test_duckdb_real_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """

    def mock_agentic_loop(*_args: object, **_kwargs: object) -> dict:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return {"final_sql": "SELECT COUNT(*) FROM test", "results": [(1,)], "success": True}

    ext = __import__("gemma_4_sql.sdk.duckdb_extension", fromlist=[""])
    monkeypatch.setattr(ext, "run_agentic_loop", mock_agentic_loop)
    conn = duckdb.connect()
    conn.execute("CREATE TABLE test (a INT); INSERT INTO test VALUES (1);")
    embed_in_duckdb(conn, model_name="mock", backend="jax")
    res = conn.execute("SELECT ask_gemma('How many tests?')").fetchall()
    val = res[0][0]
    parsed = json.loads(val)
    if parsed["success"] is not True:
        raise AssertionError
    if parsed["generated_sql"] != "SELECT COUNT(*) FROM test":
        raise AssertionError
    if parsed["results"] != [[1]]:
        raise AssertionError
