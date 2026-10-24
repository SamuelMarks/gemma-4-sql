"""DuckDB extension module for Gemma 4."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.lazy_loader import LazyLoader
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.agent import AgentContext, run_agentic_loop

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import AgentLoopResultDict

logger = logging.getLogger(__name__)
duckdb = LazyLoader("duckdb").get_module()


def _extract_duckdb_ddl(cur: Any) -> str:
    """Extract CREATE TABLE DDL statements from DuckDB information_schema.

    Args:
        cur: Active DuckDB connection or cursor.

    Returns:
        Concatenated DDL string for all tables in the `'main'` schema.

    """
    tables = cur.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'",
    ).fetchall()
    ddl_parts: list[str] = []
    for (t,) in tables:
        cols = cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ?",
            [t],
        ).fetchall()
        col_defs = ", ".join(f"{c[0]} {c[1]}" for c in cols)
        ddl_parts.append(f"CREATE TABLE {t} ({col_defs});")
    return chr(10).join(ddl_parts)


def embed_in_duckdb(
    conn: object,
    model_name: str,
    backend: str = "jax",
    db_path: str = ":memory:",
    max_retries: int = 3,
    *,
    ddl: str | None = None,
) -> None:
    """Register a scalar function in DuckDB to ask natural language questions.

    The function 'ask_gemma' will take a natural language string, use the Gemma 4 model
    to generate the appropriate SQL, and return the execution results as a JSON string.

    Args:
        conn: The database connection object.
        model_name: The name of the target model.
        backend: The backend framework to use.
        db_path: The file path to the database.
        max_retries: The integer value for max retries.
        ddl: Optional pre-extracted schema DDL to avoid repeated schema introspection.

    Raises:
        DependencyMissingError: If `duckdb` is missing.

    """
    if duckdb is None:
        msg = "duckdb is required. Install with `pip install duckdb`."
        raise DependencyMissingError(msg, dependency_name="duckdb", backend="duckdb")

    duck_conn: Any = conn
    cached_ddl: str | None = ddl

    def ask_gemma(prompt: str) -> str:
        """Execute a self-correction loop to translate prompt to SQL, run it, and return results.

        Args:
            prompt: The natural language prompt to translate to SQL.

        Returns:
            A JSON string containing the generated SQL, execution results, and success status.

        """
        nonlocal cached_ddl
        try:
            cur = duck_conn.cursor() if callable(getattr(duck_conn, "cursor", None)) else duck_conn
            if cached_ddl is None:
                cached_ddl = _extract_duckdb_ddl(cur)
            context = AgentContext(db_path=db_path, ddl=cached_ddl, db_type="duckdb", max_retries=max_retries)
            loop_res = run_agentic_loop(
                model_name=model_name,
                prompt=prompt,
                backend=backend,
                context=context,
                db_kwargs={"existing_conn": cur},
            )
            res: AgentLoopResultDict = loop_res[0] if isinstance(loop_res, list) else loop_res
            return json.dumps({
                "generated_sql": res.get("final_sql", ""),
                "results": res.get("results", []),
                "success": res.get("success", False),
            })
        except Exception as e:
            logger.exception("DuckDB UDF ask_gemma failed: ")
            return json.dumps({
                "error": str(e),
                "generated_sql": "",
                "results": [],
                "success": False,
            })

    duck_conn.create_function("ask_gemma", ask_gemma, [str], str)


__all__ = ["embed_in_duckdb"]
