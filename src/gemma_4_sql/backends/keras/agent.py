"""Keras-specific agentic self-correction loop."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from gemma_4_sql.backends.keras.inference import generate_sql
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue


async def _process_single_prompt(model_name: str, prompt: str, engine: LiveDatabaseEngine, max_retries: int, min_confidence: float) -> JSONDict:
    current_prompt = prompt
    attempts = 0
    success = False
    final_sql = ""
    history: list[JSONDict] = []

    while attempts < max_retries:
        attempts += 1
        gen_res = generate_sql(model_name, current_prompt)
        sql = gen_res.get("sql", "")
        confidence_score = float(gen_res.get("confidence_score", 1.0))

        if min_confidence and confidence_score < min_confidence:
            history.append({"attempt": attempts, "prompt": current_prompt, "sql": sql, "success": False, "error": f"Confidence score {confidence_score:.2f} below threshold {min_confidence}"})
            current_prompt = f"{prompt}\nPrevious attempt was rejected due to low confidence ({confidence_score:.2f}). Please provide a more certain SQL query."
            final_sql = sql  # type: ignore[assignment]
            continue

        (is_success, _, error_msg) = await engine.execute_with_feedback_async(sql)  # type: ignore[arg-type]
        error_msg = error_msg[:500] + "... (truncated)" if error_msg and len(error_msg) > 500 else error_msg
        history.append({"attempt": attempts, "prompt": current_prompt, "sql": sql, "success": is_success, "error": error_msg})
        if is_success:
            success = True
            final_sql = sql  # type: ignore[assignment]
            break
        current_prompt = f"{prompt}\nPrevious attempt failed with error: {error_msg}\nPlease fix the SQL query."
        final_sql = sql  # type: ignore[assignment]

    return {"backend": "keras", "model": model_name, "initial_prompt": prompt, "final_sql": final_sql, "success": success, "attempts": attempts, "history": history, "status": "completed"}


def run_agentic_loop(model_name: str, prompt: str | list[str], db_path: str = ":memory:", ddl: str | None = None, db_type: str = "sqlite", **kwargs: JSONValue) -> JSONDict | list[JSONDict]:
    """Run an agentic self-correction loop using the Keras backend.

    Args:
    ----
        model_name: The name of the model to use.
        prompt: The initial natural language prompt or a list of prompts.
        db_path: Database connection string/path.
        ddl: Optional schema setup.
        db_type: Database type.
        **kwargs: Additional DB connection kwargs.

    Returns:
    -------
        A dictionary containing the final SQL, status, and feedback history. Or a list of such dictionaries if multiple prompts were provided.

    """
    db_kwargs = kwargs.get("db_kwargs")
    max_retries = int(kwargs.get("max_retries", 3))  # type: ignore[arg-type]
    min_confidence = float(kwargs.get("min_confidence", 0.0))  # type: ignore[arg-type]
    engine = LiveDatabaseEngine(db_path=db_path, ddl=ddl, db_type=db_type, db_kwargs=db_kwargs)

    prompts = prompt if isinstance(prompt, list) else [prompt]

    async def _run_all() -> list[JSONDict]:
        tasks = [_process_single_prompt(model_name, p, engine, max_retries, min_confidence) for p in prompts]
        return await asyncio.gather(*tasks)

    try:
        results = asyncio.run(_run_all())
    finally:
        engine.close()

    return results if isinstance(prompt, list) else results[0]
