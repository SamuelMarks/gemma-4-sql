"""Real DuckDB integration tests for Gemma 4 DuckDB extension."""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from gemma_4_sql.sdk.duckdb_extension import embed_in_duckdb


class MockBackendForDuckDB:
    """Mock backend that returns a query matching the prompt."""

    def generate_sql(self, _model_name: str, _prompt: str, **_kwargs: object) -> dict[str, Any]:
        """Generate SQL query.

        Args:
            _model_name: Target model name.
            _prompt: Input prompt.
            **_kwargs: Optional kwargs.

        Returns:
            Dictionary with generated SQL.
        """
        return {"sql": "SELECT COUNT(*) FROM test;", "confidence_score": 0.95}


def test_duckdb_real_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test real DuckDB in-memory database integration with ask_gemma UDF.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    real_duckdb = getattr(pytest, "_real_duckdb", None)
    if real_duckdb is None:
        try:
            import duckdb as real_duckdb
        except ImportError:
            pytest.skip("DuckDB is not installed")

    monkeypatch.setitem(sys.modules, "duckdb", real_duckdb)
    monkeypatch.setattr("gemma_4_sql.sdk.duckdb_extension.duckdb", real_duckdb)
    monkeypatch.setattr("gemma_4_sql.sdk.adapters.duckdb_adapter.duckdb", real_duckdb)

    # Register mock backend
    reg = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"])
    monkeypatch.setattr(reg, "get_backend", lambda _name: MockBackendForDuckDB())

    conn = real_duckdb.connect(":memory:")
    conn.execute("CREATE TABLE test (a INT);")
    conn.execute("INSERT INTO test VALUES (1), (2), (3);")

    embed_in_duckdb(conn, model_name="test-model", backend="test-backend", db_path=":memory:")

    # Execute UDF inside DuckDB
    res = conn.execute("SELECT ask_gemma('How many tests?')").fetchall()
    assert len(res) == 1
    val = str(res[0][0])
    parsed = json.loads(val)

    assert parsed["success"] is True
    assert "COUNT(*)" in parsed["generated_sql"]
    assert parsed["results"] == [[3]]
    conn.close()


def test_duckdb_udf_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ask_gemma error handling when query execution fails.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    real_duckdb = getattr(pytest, "_real_duckdb", None)
    if real_duckdb is None:
        try:
            import duckdb as real_duckdb
        except ImportError:
            pytest.skip("DuckDB is not installed")

    class BadBackend:
        """Backend that raises an error."""

        def generate_sql(self, _model_name: str, _prompt: str, **_kwargs: object) -> dict[str, Any]:
            """Raise RuntimeError.

            Args:
                _model_name: Target model.
                _prompt: Prompt.
                **_kwargs: Kwargs.

            Raises:
                RuntimeError: Backend failure.
            """
            raise RuntimeError("Model generation failed")

    monkeypatch.setitem(sys.modules, "duckdb", real_duckdb)
    monkeypatch.setattr("gemma_4_sql.sdk.duckdb_extension.duckdb", real_duckdb)
    monkeypatch.setattr("gemma_4_sql.sdk.adapters.duckdb_adapter.duckdb", real_duckdb)

    reg = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"])
    monkeypatch.setattr(reg, "get_backend", lambda _name: BadBackend())

    conn = real_duckdb.connect(":memory:")
    embed_in_duckdb(conn, model_name="bad-model", backend="test-backend")

    res = conn.execute("SELECT ask_gemma('invalid')").fetchall()
    assert len(res) == 1
    parsed = json.loads(str(res[0][0]))
    assert parsed["success"] is False
    assert "error" in parsed
    conn.close()
