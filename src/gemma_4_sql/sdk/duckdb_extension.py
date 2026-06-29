"""DuckDB extension module for Gemma 4."""

from __future__ import annotations

import json

from gemma_4_sql.sdk.agent import run_agentic_loop

try:
    import duckdb
except ImportError:
    duckdb = None


def embed_in_duckdb(conn: object, model_name: str, backend: str = "jax", db_path: str = ":memory:", max_retries: int = 3) -> None:
    """Register a scalar function in DuckDB to ask natural language questions.

    The function 'ask_gemma' will take a natural language string, use the
    Gemma 4 model to generate the appropriate SQL, and return the execution
    results as a JSON string.

    Args:
    ----
        conn: The DuckDB connection.
        model_name: The name or path of the model.
        backend: The backend framework ('jax', 'keras', 'maxtext', 'pytorch').
        db_path: The database path for the agent to connect to (if not :memory:).
        max_retries: Max number of execution attempts.

    """
    if duckdb is None:
        msg = "duckdb is required. Install with `pip install duckdb`."
        raise ImportError(msg)

    def ask_gemma(prompt: str) -> str:
        """Execute a self-correction loop to translate prompt to SQL, run it, and return results.

        Args:
        ----
            prompt: The natural language prompt to translate to SQL.

        Returns:
        -------
            A JSON string containing the generated SQL, execution results, and success status.

        """
        tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()  # type: ignore[attr-defined]
        ddl_parts = []
        for (t,) in tables:
            cols = conn.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{t}'".replace("{t}", t)).fetchall()  # type: ignore[attr-defined]
            col_defs = ", ".join(f"{c[0]} {c[1]}" for c in cols)
            ddl_parts.append(f"CREATE TABLE {t} ({col_defs});")
        ddl = "\n".join(ddl_parts)
        res = run_agentic_loop(model_name=model_name, prompt=prompt, backend=backend, db_path=db_path, ddl=ddl, db_type="duckdb", max_retries=max_retries)
        return json.dumps({"generated_sql": res.get("final_sql", ""), "results": res.get("results", []), "success": res.get("success", False)})

    conn.create_function("ask_gemma", ask_gemma, [str], str)  # type: ignore[attr-defined]
