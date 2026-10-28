"""Real DuckDB integration tests for Gemma 4 DuckDB extension."""

from __future__ import annotations

import json
from typing import NoReturn

import duckdb
import pytest

from gemma_4_sql.sdk.duckdb_extension import embed_in_duckdb
from gemma_4_sql.type_hints import InferenceResultDict, JSONValue
from tests.conftest import requires_duckdb

pytestmark = requires_duckdb


class MockBackendForDuckDB:
    """Mock backend that verifies schema DDL introspection and returns a count query against `test`."""

    def generate_sql(self, model_name: str, prompt: str, **_kwargs: JSONValue) -> InferenceResultDict:
        """Generate SQL query against the introspected `test` table.

        Args:
            model_name: Target model name.
            prompt: Input prompt containing RAG schema context.
            **_kwargs: Optional kwargs.

        Returns:
            Dictionary with generated SQL.

        """
        assert "Table: test" in prompt
        return {
            "backend": "test-backend",
            "model": model_name,
            "prompt": prompt,
            "sql": "SELECT COUNT(*) FROM test;",
            "beam_width": 3,
            "confidence_score": 0.95,
            "status": "success",
        }


def test_duckdb_real_integration(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    """Test real DuckDB in-memory database integration with ask_gemma UDF.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        in_memory_duckdb: Shared in-memory DuckDB connection fixture.

    """
    import duckdb as real_duckdb

    import gemma_4_sql.sdk.agent as ag

    monkeypatch.setattr("gemma_4_sql.sdk.duckdb_extension.duckdb", real_duckdb)
    monkeypatch.setattr("gemma_4_sql.sdk.adapters.duckdb_adapter.duckdb", real_duckdb)
    monkeypatch.setattr(ag, "get_backend", lambda _name: MockBackendForDuckDB())  # pyright: ignore[reportUnknownLambdaType]

    in_memory_duckdb.execute("CREATE TABLE test (a INT);")
    in_memory_duckdb.execute("INSERT INTO test VALUES (1), (2), (3);")

    embed_in_duckdb(in_memory_duckdb, model_name="test-model", backend="test-backend", db_path=":memory:")

    res = in_memory_duckdb.execute("SELECT ask_gemma('How many tests?')").fetchall()
    assert len(res) == 1
    val = str(res[0][0])
    parsed = json.loads(val)

    assert parsed["success"] is True
    assert parsed["generated_sql"] == "SELECT COUNT(*) FROM test;"
    assert parsed["results"] == [[3]]


def test_duckdb_udf_error_handling(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    """Test ask_gemma error handling when query execution fails.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        in_memory_duckdb: Shared in-memory DuckDB connection fixture.

    """
    import duckdb as real_duckdb

    import gemma_4_sql.sdk.agent as ag

    class BadBackend:
        """Backend that raises an error."""

        def generate_sql(self, _model_name: str, _prompt: str, **_kwargs: JSONValue) -> NoReturn:
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
    monkeypatch.setattr(ag, "get_backend", lambda _name: BadBackend())  # pyright: ignore[reportUnknownLambdaType]

    embed_in_duckdb(in_memory_duckdb, model_name="bad-model", backend="test-backend")

    res = in_memory_duckdb.execute("SELECT ask_gemma('invalid')").fetchall()
    assert len(res) == 1
    parsed = json.loads(str(res[0][0]))
    assert parsed["success"] is False
    assert "error" in parsed
