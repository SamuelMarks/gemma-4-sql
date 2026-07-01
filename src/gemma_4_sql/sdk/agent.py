"""SDK Agent module for self-correction execution feedback loops."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue


def run_agentic_loop(model_name: str, prompt: str | list[str], backend: str = "jax", db_path: str = ":memory:", ddl: str | None = None, **kwargs: JSONValue) -> JSONDict | list[JSONDict]:
    """Run an agentic self-correction loop.

    Args:
    ----
        model_name: The name or path of the model.
        prompt: The natural language prompt.
        backend: The backend framework ('jax', 'keras', 'maxtext', 'pytorch').
        db_path: Path to the SQLite database for execution accuracy.
        ddl: Optional DDL to set up the schema.
        db_type: Type of database engine ('sqlite', 'postgresql', 'snowflake').
        db_kwargs: Additional keyword arguments for DB engine connection.
        max_retries: Max number of attempts.
        **kwargs: Additional keyword arguments.

    Returns:
    -------
        Agentic loop results dictionary.

    """
    db_type = kwargs.get("db_type", "sqlite")
    db_kwargs = kwargs.get("db_kwargs")
    max_retries = kwargs.get("max_retries", 3)
    min_confidence = kwargs.get("min_confidence", 0.0)
    kwargs = {"model_name": model_name, "prompt": prompt, "db_path": db_path, "ddl": ddl, "db_type": db_type, "db_kwargs": db_kwargs, "max_retries": max_retries, "min_confidence": min_confidence}
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).run_agentic_loop(**kwargs)
