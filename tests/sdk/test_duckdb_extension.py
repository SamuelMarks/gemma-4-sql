"""Tests for `duckdb_extension.py` using real DuckDB and strongly-typed protocol doubles."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from unittest.mock import patch

import pytest

import gemma_4_sql.sdk.duckdb_extension as de
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.agent import AgentContext
from gemma_4_sql.sdk.duckdb_extension import DuckDBCursorProtocol, embed_in_duckdb
from gemma_4_sql.type_hints import AgentLoopResultDict, JSONPrimitive, JSONValue
from tests.conftest import requires_duckdb


@requires_duckdb
def test_embed_in_duckdb_success_and_late_table_ddl() -> None:
    """Verify `embed_in_duckdb` snapshots existing DDL AND lazily extracts DDL when tables are created later."""
    import duckdb

    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER); INSERT INTO users VALUES (42);")

    embed_in_duckdb(conn, "model", "pytorch")

    captured_prompts: list[str] = []

    def _fake_agent_loop(
        model_name: str,
        prompt: str,
        backend: str = "jax",
        context: AgentContext | None = None,
        **_kwargs: JSONValue | dict[str, DuckDBCursorProtocol],
    ) -> AgentLoopResultDict:
        """Record prompt and return deterministic agent loop result.

        Returns:
            Simulated `AgentLoopResultDict` payload.

        """
        _ = (model_name, backend, context)
        captured_prompts.append(prompt)
        return {
            "backend": backend,
            "initial_prompt": prompt,
            "final_sql": "SELECT * FROM users",
            "results": [(42,)],
            "success": True,
            "attempts": 1,
            "history": [],
            "status": "completed",
        }

    with patch.object(de, "run_agentic_loop", side_effect=_fake_agent_loop):
        row1 = conn.execute("SELECT ask_gemma('Get users')").fetchone()
        assert row1 is not None
        res_json = json.loads(str(row1[0]))
        assert res_json["success"] is True
        assert res_json["generated_sql"] == "SELECT * FROM users"
        assert res_json["results"] == [[42]]
        assert len(captured_prompts) == 1
        assert "Table: users" in captured_prompts[0]

        row2 = conn.execute("SELECT ask_gemma('Get users again')").fetchone()
        assert row2 is not None
        assert json.loads(str(row2[0]))["success"] is True

        # Also test explicit `ddl=...` parameter (skipping initial snapshot)
        conn_explicit = duckdb.connect(":memory:")
        conn_explicit.execute("CREATE TABLE explicit_tbl (id INT);")
        embed_in_duckdb(conn_explicit, "model", "pytorch", ddl="CREATE TABLE explicit_tbl (id INT);")
        row_exp = conn_explicit.execute("SELECT ask_gemma('Count explicit')").fetchone()
        assert row_exp is not None
        assert json.loads(str(row_exp[0]))["success"] is True
        conn_explicit.close()

        # Also test registering BEFORE tables exist (snapshot empty -> lazy extract inside UDF)
        conn_late = duckdb.connect(":memory:")
        embed_in_duckdb(conn_late, "model", "pytorch")
        conn_late.execute("CREATE TABLE orders (order_id INT);")
        row_late = conn_late.execute("SELECT ask_gemma('Count orders')").fetchone()
        assert row_late is not None
        assert json.loads(str(row_late[0]))["success"] is True
        assert "Table: orders" in captured_prompts[-1]
        conn_late.close()

    conn.close()


def test_embed_in_duckdb_missing_and_identifier_quoting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing `duckdb` error, `information_schema` fallback quoting, and non-cursor connection support."""

    class _MinimalResult:
        """Minimal `DuckDBCursorProtocol` result wrapper."""

        def __init__(self, rows: Sequence[Sequence[JSONPrimitive]]) -> None:
            """Store rows."""
            self._rows = rows

        def execute(self, _query: str, _parameters: Sequence[JSONPrimitive] | None = None, /) -> DuckDBCursorProtocol:
            """Return `self`."""
            return self

        def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
            """Return stored rows."""
            return self._rows

    class _FallbackSchemaConn:
        """Typed connection without `.cursor()` exercising `information_schema` fallback and identifier quoting."""

        def __init__(self, *, fail_all_queries: bool = False, empty_schema: bool = False) -> None:
            """Configure fallback behavior."""
            self._fail_all = fail_all_queries
            self._empty = empty_schema
            self.registered_fn: Callable[[str], str] | None = None

        def execute(self, query: str, parameters: Sequence[JSONPrimitive] | None = None, /) -> DuckDBCursorProtocol:
            """Simulate `duckdb_tables()` failure and `information_schema` responses.

            Returns:
                `_MinimalResult` cursor wrapping simulated catalog rows.

            Raises:
                RuntimeError: When `_fail_all` is set or `duckdb_tables()` is queried.

            """
            if self._fail_all:
                raise RuntimeError("catalog unavailable")
            if "duckdb_tables()" in query:
                raise RuntimeError("duckdb_tables not supported")
            if self._empty:
                return _MinimalResult([])
            if "information_schema.tables" in query:
                return _MinimalResult([("users",), ('my "special" table',), ("drop",)])
            assert parameters is not None
            t_name = str(parameters[0])
            if t_name == "users":
                return _MinimalResult([("id", "INTEGER"), ("name", "VARCHAR")])
            return _MinimalResult([("select", "INTEGER"), ("total amount", "DOUBLE")])

        def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
            """Return empty rows."""
            return []

        def create_function(self, name: str, function: Callable[..., str]) -> None:
            """Capture registered UDF."""
            _ = name
            self.registered_fn = function

        def close(self) -> None:
            """No-op close."""

    with pytest.raises(DependencyMissingError, match="duckdb is required"):
        de.embed_in_duckdb(_FallbackSchemaConn(), "model", "jax", duckdb_module=None)

    fallback_conn = _FallbackSchemaConn()
    assert fallback_conn.fetchall() == []
    assert _MinimalResult([]).execute("SELECT 1").fetchall() == []
    ddl_str = de._extract_duckdb_ddl(fallback_conn)
    assert "CREATE TABLE users (id INTEGER, name VARCHAR);" in ddl_str
    assert 'CREATE TABLE "my ""special"" table" ("select" INTEGER, "total amount" DOUBLE);' in ddl_str
    assert 'CREATE TABLE "drop" ("select" INTEGER, "total amount" DOUBLE);' in ddl_str

    assert de._extract_duckdb_ddl(_FallbackSchemaConn(empty_schema=True)) == ""

    # Exercise `embed_in_duckdb` when initial snapshot raises RuntimeError AND connection has no `.cursor()`
    monkeypatch.setattr(de, "duckdb", de._LoaderSentinel.UNRESOLVED)
    err_conn = _FallbackSchemaConn(fail_all_queries=True)
    de.embed_in_duckdb(err_conn, "model", "jax")
    assert err_conn.registered_fn is not None
    err_payload = json.loads(err_conn.registered_fn("Find users"))
    assert err_payload["success"] is False
    assert "catalog unavailable" in err_payload["error"]
