"""DuckDB extension module for Gemma 4."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from gemma_4_sql.backends.lazy_loader import LazyLoader
from gemma_4_sql.sdk.agent import AgentContext, run_agentic_loop

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict

duckdb = LazyLoader("duckdb").get_module()


def embed_in_duckdb(
    conn: object,
    model_name: str,
    backend: str = "jax",
    db_path: str = ":memory:",
    max_retries: int = 3,
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

    Raises:
        ImportError: If duckdb is missing.
    """
    if duckdb is None:
        msg = "duckdb is required. Install with `pip install duckdb`."
        raise ImportError(msg)

    def ask_gemma(prompt: str) -> str:
        """Execute a self-correction loop to translate prompt to SQL, run it, and return results.

        Args:
            prompt: The natural language prompt to translate to SQL.

        Returns:
            A JSON string containing the generated SQL, execution results, and success status.
        """
        duck_conn = cast(Any, conn)
        tables = duck_conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
        ddl_parts = []
        for (t,) in tables:
            cols = duck_conn.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ?",
                [t],
            ).fetchall()
            col_defs = ", ".join(f"{c[0]} {c[1]}" for c in cols)
            ddl_parts.append(f"CREATE TABLE {t} ({col_defs});")
        ddl = "\n".join(ddl_parts)
        context = AgentContext(db_path=db_path, ddl=ddl, db_type="duckdb", max_retries=max_retries)
        loop_res = run_agentic_loop(model_name=model_name, prompt=prompt, backend=backend, context=context)
        res: JSONDict = loop_res[0] if isinstance(loop_res, list) else loop_res
        return json.dumps({
            "generated_sql": res.get("final_sql", ""),
            "results": res.get("results", []),
            "success": res.get("success", False),
        })

    duck_conn = cast(Any, conn)
    duck_conn.create_function("ask_gemma", ask_gemma, [str], str)
