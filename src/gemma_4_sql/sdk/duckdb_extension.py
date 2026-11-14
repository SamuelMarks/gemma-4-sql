"""DuckDB extension module for Gemma 4."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.lazy_loader import LazyLoader, ModuleNamespaceProtocol
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.agent import AgentContext, run_agentic_loop
from gemma_4_sql.sdk.rag import build_rag_prompt
from gemma_4_sql.sdk.sql_lexer import DANGEROUS_KEYWORDS
from gemma_4_sql.type_hints import (
    AgentLoopResultDict,
    CursorProtocol,
    DatabaseConnectionProtocol,
    DuckDBConnectionProtocol,
    JSONPrimitive,
)

logger = logging.getLogger(__name__)
_SIMPLE_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_RESERVED_SQL_IDENTIFIERS: frozenset[str] = DANGEROUS_KEYWORDS | frozenset({
    "ALL",
    "AND",
    "AS",
    "BY",
    "CASE",
    "CHECK",
    "DEFAULT",
    "DISTINCT",
    "ELSE",
    "END",
    "FOREIGN",
    "FROM",
    "GROUP",
    "HAVING",
    "IN",
    "INDEX",
    "INNER",
    "INTO",
    "IS",
    "JOIN",
    "KEY",
    "LEFT",
    "LIMIT",
    "NOT",
    "NULL",
    "OFFSET",
    "ON",
    "OR",
    "ORDER",
    "PRIMARY",
    "REFERENCES",
    "RIGHT",
    "SELECT",
    "TABLE",
    "THEN",
    "UNION",
    "UNIQUE",
    "VALUES",
    "WHEN",
    "WHERE",
    "WITH",
})
_NL = """
"""


class _LoaderSentinel(Enum):
    """Sentinel indicating `duckdb` has not yet been lazily imported."""

    UNRESOLVED = "unresolved"


_DUCKDB_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("duckdb")
duckdb: ModuleNamespaceProtocol | _LoaderSentinel | None = _LoaderSentinel.UNRESOLVED


def _resolve_duckdb(
    override: ModuleNamespaceProtocol | _LoaderSentinel | None = _LoaderSentinel.UNRESOLVED,
) -> ModuleNamespaceProtocol | None:
    """Resolve `duckdb` lazily while honoring explicit DI or module-attribute overrides.

    Args:
        override: Optional explicitly injected `duckdb` module or `None`.

    Returns:
        Loaded `duckdb` module namespace or `None`.

    """
    global duckdb
    if override is not _LoaderSentinel.UNRESOLVED:
        return override
    if duckdb is _LoaderSentinel.UNRESOLVED:
        duckdb = _DUCKDB_LOADER.get_module()
    return duckdb


@runtime_checkable
class DuckDBCursorProtocol(Protocol):
    """Protocol for DuckDB connections and cursors supporting `.execute(...).fetchall()`."""

    def execute(self, query: str, parameters: Sequence[JSONPrimitive] | None = ..., /) -> DuckDBCursorProtocol:
        """Execute a SQL statement and return a fetchable cursor.

        Args:
            query: SQL statement text.
            parameters: Optional positional bind parameters.

        Returns:
            Fetchable `DuckDBCursorProtocol` cursor.

        """
        ...

    def fetchall(self) -> Sequence[Sequence[JSONPrimitive]]:
        """Fetch all result rows.

        Returns:
            Sequence of result rows.

        """
        ...


@runtime_checkable
class _CursorSpawningConnProtocol(Protocol):
    """Protocol for DuckDB connections exposing `.cursor()` to isolate UDF execution."""

    def cursor(self) -> DuckDBCursorProtocol:
        """Return a duplicate DuckDB cursor.

        Returns:
            Duplicate `DuckDBCursorProtocol` cursor.

        """
        ...


@runtime_checkable
class DuckDBMinimalUDFConnProtocol(DuckDBCursorProtocol, Protocol):
    """Protocol for minimal DuckDB connections supporting `.execute()` and `.create_function(...)`."""

    def create_function(
        self,
        name: str,
        function: Callable[..., str],
    ) -> DuckDBCursorProtocol | None:
        """Register a Python scalar UDF with DuckDB.

        Args:
            name: SQL function identifier.
            function: Python callable invoked per row.

        Returns:
            Optional connection/cursor handle.

        """
        ...


@runtime_checkable
class DuckDBUDFConnProtocol(DuckDBMinimalUDFConnProtocol, _CursorSpawningConnProtocol, Protocol):
    """Protocol for DuckDB connections supporting both `.cursor()` and `.create_function(...)`."""


type DuckDBEmbeddableConnection = DuckDBConnectionProtocol | DuckDBUDFConnProtocol | DuckDBMinimalUDFConnProtocol


def _quote_duckdb_ident(ident: str) -> str:
    """Quote a DuckDB table or column identifier when it contains special characters or reserved keywords.

    Args:
        ident: Raw identifier string from `information_schema`.

    Returns:
        Safe unquoted identifier or double-quoted SQL identifier with doubled internal quotes.

    """
    if _SIMPLE_IDENT_RE.match(ident) and ident.upper() not in _RESERVED_SQL_IDENTIFIERS:
        return ident
    escaped = ident.replace('"', '""')
    return f'"{escaped}"'


def _extract_duckdb_ddl(cur: DuckDBCursorProtocol | DatabaseConnectionProtocol | CursorProtocol) -> str:
    """Extract CREATE TABLE DDL statements from DuckDB metadata or information_schema.

    Queries `duckdb_tables()` first to preserve full `CREATE TABLE` DDL including
    constraints, primary keys, and defaults, falling back to `information_schema`
    with safe identifier quoting when `duckdb_tables()` is unavailable or empty.

    Args:
        cur: Active DuckDB connection or cursor.

    Returns:
        Concatenated DDL string for all user tables in the `'main'` schema.

    """
    try:
        exec_res = cur.execute(
            "SELECT sql FROM duckdb_tables() WHERE schema_name = 'main' AND internal = false AND sql IS NOT NULL",
        )
        rows = exec_res.fetchall()
        sql_defs = [str(r[0]).strip() for r in rows if r and r[0]]
        if sql_defs:
            return _NL.join(s if s.endswith(";") else f"{s};" for s in sql_defs)
    except (RuntimeError, ValueError, TypeError, AttributeError, OSError):
        pass

    exec_tables = cur.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'",
    )
    tables = exec_tables.fetchall()
    ddl_parts: list[str] = []
    for (t,) in tables:
        t_str = str(t)
        exec_cols = cur.execute(
            (
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'main' AND table_name = ?"
            ),
            [t_str],
        )
        cols = exec_cols.fetchall()
        col_defs = ", ".join(f"{_quote_duckdb_ident(str(c[0]))} {c[1]}" for c in cols)
        ddl_parts.append(f"CREATE TABLE {_quote_duckdb_ident(t_str)} ({col_defs});")
    return _NL.join(ddl_parts)


def _spawn_udf_cursor(
    duck_conn: DuckDBEmbeddableConnection,
) -> DuckDBEmbeddableConnection | DatabaseConnectionProtocol | DuckDBCursorProtocol | CursorProtocol:
    """Create an isolated duplicate cursor when supported by `duck_conn` to avoid connection re-entrancy.

    Args:
        duck_conn: Host DuckDB connection.

    Returns:
        Duplicate cursor if `.cursor()` is available, otherwise `duck_conn`.

    """
    if isinstance(duck_conn, _CursorSpawningConnProtocol) and callable(duck_conn.cursor):
        return duck_conn.cursor()
    return duck_conn


def embed_in_duckdb(
    conn: DuckDBEmbeddableConnection,
    model_name: str,
    backend: str = "jax",
    db_path: str = ":memory:",
    max_retries: int = 3,
    *,
    ddl: str | None = None,
    duckdb_module: ModuleNamespaceProtocol | _LoaderSentinel | None = _LoaderSentinel.UNRESOLVED,
) -> None:
    """Register a scalar function in DuckDB to ask natural language questions.

    Snapshots schema DDL ahead of UDF execution when tables already exist so `ask_gemma`
    avoids unnecessary catalog introspection inside active query execution.

    Args:
        conn: The database connection object.
        model_name: The name of the target model.
        backend: The backend framework to use.
        db_path: The file path to the database.
        max_retries: Maximum number of self-correction attempts.
        ddl: Optional pre-extracted schema DDL to avoid repeated schema introspection.
        duckdb_module: Optional injected `duckdb` module namespace or `None` for testing.

    Raises:
        DependencyMissingError: If `duckdb` is missing.

    """
    if _resolve_duckdb(duckdb_module) is None:
        msg = "duckdb is required. Install with `pip install duckdb`."
        raise DependencyMissingError(msg, dependency_name="duckdb", backend="duckdb")

    duck_conn = conn
    cached_ddl: str | None = ddl
    if cached_ddl is None:
        try:
            init_cur = _spawn_udf_cursor(duck_conn)
            snap = _extract_duckdb_ddl(init_cur)
            if snap:
                cached_ddl = snap
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError):
            cached_ddl = None

    def ask_gemma(prompt: str) -> str:
        """Execute a self-correction loop to translate prompt to SQL, run it, and return results.

        Args:
            prompt: The natural language prompt to translate to SQL.

        Returns:
            A JSON string containing the generated SQL, execution results, and success status.

        """
        nonlocal cached_ddl
        try:
            cur = _spawn_udf_cursor(duck_conn)
            if cached_ddl is None:
                cached_ddl = _extract_duckdb_ddl(cur)
            augmented_prompt = build_rag_prompt(prompt, cached_ddl) if cached_ddl else prompt
            context = AgentContext(db_path=db_path, ddl=None, db_type="duckdb", max_retries=max_retries)
            res: AgentLoopResultDict = run_agentic_loop(
                model_name=model_name,
                prompt=augmented_prompt,
                backend=backend,
                context=context,
                db_kwargs={"existing_conn": cur},
            )
            return json.dumps({
                "generated_sql": res.get("final_sql", ""),
                "results": res.get("results", []),
                "success": res.get("success", False),
            })
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.exception("DuckDB UDF ask_gemma failed: ")
            return json.dumps({
                "error": str(e),
                "generated_sql": "",
                "results": [],
                "success": False,
            })

    duck_conn.create_function("ask_gemma", ask_gemma)


__all__ = ["DuckDBConnectionProtocol", "DuckDBCursorProtocol", "embed_in_duckdb"]
