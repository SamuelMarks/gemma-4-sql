"""Real DuckDB integration tests for Gemma 4 DuckDB extension."""

from __future__ import annotations

import json
from typing import Any

import pytest

from gemma_4_sql.sdk.duckdb_extension import embed_in_duckdb
from tests.conftest import requires_duckdb

pytestmark = requires_duckdb


class MockBackendForDuckDB:
    """Mock backend that returns a query matching the prompt."""

    def generate_sql(self, _model_name: str, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
        """Generate SQL query.

        Args:
            _model_name: Target model name.
            _prompt: Input prompt.
            **_kwargs: Optional kwargs.

        Returns:
            Dictionary with generated SQL.
        """
        return {"sql": "SELECT 1;", "confidence_score": 0.95}


def test_duckdb_real_integration(monkeypatch: pytest.MonkeyPatch, in_memory_duckdb: Any) -> None:
    """Test real DuckDB in-memory database integration with ask_gemma UDF.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        in_memory_duckdb: Shared in-memory DuckDB connection fixture.
    """
    import duckdb as real_duckdb

    import gemma_4_sql.sdk.agent as ag

    monkeypatch.setattr("gemma_4_sql.sdk.duckdb_extension.duckdb", real_duckdb)
    monkeypatch.setattr("gemma_4_sql.sdk.adapters.duckdb_adapter.duckdb", real_duckdb)
    monkeypatch.setattr(ag, "get_backend", lambda _name: MockBackendForDuckDB())

    in_memory_duckdb.execute("CREATE TABLE test (a INT);")
    in_memory_duckdb.execute("INSERT INTO test VALUES (1), (2), (3);")

    embed_in_duckdb(in_memory_duckdb, model_name="test-model", backend="test-backend", db_path=":memory:")

    res = in_memory_duckdb.execute("SELECT ask_gemma('How many tests?')").fetchall()
    assert len(res) == 1
    val = str(res[0][0])
    parsed = json.loads(val)

    assert "success" in parsed
    assert "1" in parsed["generated_sql"]


def test_duckdb_udf_error_handling(monkeypatch: pytest.MonkeyPatch, in_memory_duckdb: Any) -> None:
    """Test ask_gemma error handling when query execution fails.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        in_memory_duckdb: Shared in-memory DuckDB connection fixture.
    """
    import duckdb as real_duckdb

    import gemma_4_sql.sdk.agent as ag

    class BadBackend:
        """Backend that raises an error."""

        def generate_sql(self, _model_name: str, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            """Raise RuntimeError.

            Args:
                _model_name: Target model.
                _prompt: Prompt.
                **_kwargs: Kwargs.

            Raises:
                RuntimeError: Backend failure.
            """
            raise RuntimeError("Model generation failed")

    monkeypatch.setattr("gemma_4_sql.sdk.duckdb_extension.duckdb", real_duckdb)
    monkeypatch.setattr("gemma_4_sql.sdk.adapters.duckdb_adapter.duckdb", real_duckdb)
    monkeypatch.setattr(ag, "get_backend", lambda _name: BadBackend())

    embed_in_duckdb(in_memory_duckdb, model_name="bad-model", backend="test-backend")

    res = in_memory_duckdb.execute("SELECT ask_gemma('invalid')").fetchall()
    assert len(res) == 1
    parsed = json.loads(str(res[0][0]))
    assert parsed["success"] is False
    assert "error" in parsed
