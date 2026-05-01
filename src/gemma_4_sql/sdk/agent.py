"""
SDK Agent module for self-correction execution feedback loops.
"""

from __future__ import annotations

from typing import Any


def run_agentic_loop(
    model_name: str,
    prompt: str,
    backend: str = "jax",
    db_path: str = ":memory:",
    ddl: str | None = None,
    db_type: str = "sqlite",
    db_kwargs: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Runs an agentic self-correction loop.

    Args:
        model_name: The name or path of the model.
        prompt: The natural language prompt.
        backend: The backend framework ('jax', 'keras', 'maxtext', 'pytorch').
        db_path: Path to the SQLite database for execution accuracy.
        ddl: Optional DDL to set up the schema.
        db_type: Type of database engine ('sqlite', 'postgresql', 'snowflake').
        db_kwargs: Additional keyword arguments for DB engine connection.
        max_retries: Max number of attempts.

    Returns:
        Agentic loop results dictionary.
    """
    kwargs = {
        "model_name": model_name,
        "prompt": prompt,
        "db_path": db_path,
        "ddl": ddl,
        "db_type": db_type,
        "db_kwargs": db_kwargs,
        "max_retries": max_retries,
    }

    from gemma_4_sql.sdk.registry import get_backend
    return get_backend(backend).run_agentic_loop(**kwargs)
