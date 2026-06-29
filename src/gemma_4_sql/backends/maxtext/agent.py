"""MaxText-specific agentic self-correction loop."""

from __future__ import annotations

from gemma_4_sql.backends.maxtext.inference import generate_sql
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine


def run_agentic_loop(model_name: str, prompt: str, db_path: str = ":memory:", ddl: str | None = None, db_type: str = "sqlite", **kwargs: object) -> dict[str, object]:
    """Run an agentic self-correction loop using the MaxText backend.

    Args:
    ----
        model_name: The name of the model to use.
        prompt: The initial natural language prompt.
        db_path: Database connection string/path.
        ddl: Optional schema setup.
        db_type: Database type.
        db_kwargs: Additional DB connection kwargs.
        max_retries: Max number of self-correction attempts.
        **kwargs: Additional keyword arguments.

    Returns:
    -------
        A dictionary containing the final SQL, status, and feedback history.

    """
    db_kwargs = kwargs.get("db_kwargs")
    max_retries = kwargs.get("max_retries", 3)
    engine = LiveDatabaseEngine(db_path=db_path, ddl=ddl, db_type=db_type, db_kwargs=db_kwargs)
    current_prompt = prompt
    attempts = 0
    success = False
    final_sql = ""
    history: list[dict[str, object]] = []
    try:
        while attempts < max_retries:  # type: ignore[operator]
            attempts += 1
            gen_res = generate_sql(model_name, current_prompt)
            sql = gen_res.get("sql", "")
            (is_success, _, error_msg) = engine.execute_with_feedback(sql)  # type: ignore[arg-type]
            history.append({"attempt": attempts, "prompt": current_prompt, "sql": sql, "success": is_success, "error": error_msg})
            if is_success:
                success = True
                final_sql = sql  # type: ignore[assignment]
                break
            current_prompt = f"{prompt}\nPrevious attempt failed with error: {error_msg}\nPlease fix the SQL query."
            final_sql = sql  # type: ignore[assignment]
    finally:
        engine.close()
    return {"backend": "maxtext", "model": model_name, "initial_prompt": prompt, "final_sql": final_sql, "success": success, "attempts": attempts, "history": history, "status": "completed"}
