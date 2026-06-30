import json

import duckdb
import pytest

from gemma_4_sql.sdk.duckdb_extension import embed_in_duckdb


def test_duckdb_real_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch the agentic loop to not actually call a real LLM
    def mock_agentic_loop(*args: object, **kwargs: object) -> dict:
        return {"final_sql": "SELECT COUNT(*) FROM test", "results": [(1,)], "success": True}

    import gemma_4_sql.sdk.duckdb_extension as ext

    monkeypatch.setattr(ext, "run_agentic_loop", mock_agentic_loop)

    conn = duckdb.connect()
    conn.execute("CREATE TABLE test (a INT); INSERT INTO test VALUES (1);")

    embed_in_duckdb(conn, model_name="mock", backend="jax")

    # Actually call the UDF inside a DuckDB process query
    res = conn.execute("SELECT ask_gemma('How many tests?')").fetchall()

    # DuckDB returns a list of tuples
    val = res[0][0]
    parsed = json.loads(val)

    assert parsed["success"] is True
    assert parsed["generated_sql"] == "SELECT COUNT(*) FROM test"
    assert parsed["results"] == [[1]]
