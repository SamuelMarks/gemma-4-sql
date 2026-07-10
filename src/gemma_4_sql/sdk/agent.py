"""SDK Agent module for self-correction execution feedback loops."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine

MAX_ERR_LEN = 500
if TYPE_CHECKING:
    from gemma_4_sql.sdk.protocols import BackendProtocol
    from gemma_4_sql.type_hints import JSONDict, JSONValue


@dataclass
class AgentContext:
    """Context for the agentic loop."""

    db_path: str = ":memory:"
    ddl: str | None = None
    db_type: str = "sqlite"
    max_retries: int = 3
    min_confidence: float = 0.0


async def _process_single_prompt(backend_name: str, backend_impl: BackendProtocol, model_name: str, prompt: str, engine: LiveDatabaseEngine, context: AgentContext) -> JSONDict:
    """Execute logic.

    Returns:
        object: The resulting output from the operation.

    """
    current_prompt = prompt
    attempts = 0
    success = False
    final_sql = ""
    history: list[JSONDict] = []
    while attempts < context.max_retries:
        attempts += 1
        gen_res = backend_impl.generate_sql(model_name, current_prompt)
        sql = str(gen_res.get("sql", ""))
        confidence_score = float(gen_res.get("confidence_score", 1.0))
        if context.min_confidence and confidence_score < context.min_confidence:
            history.append({"attempt": attempts, "prompt": current_prompt, "sql": sql, "success": False, "error": f"Confidence score {confidence_score:.2f} below threshold {context.min_confidence}"})
            current_prompt = f"{prompt}\nPrevious attempt was rejected due to low confidence ({confidence_score:.2f}). Please provide a more certain SQL query."
            final_sql = sql
            continue
        (is_success, _, error_msg) = await engine.execute_with_feedback_async(sql)
        error_msg = error_msg[:MAX_ERR_LEN] + "... (truncated)" if error_msg and len(error_msg) > MAX_ERR_LEN else error_msg
        history.append({"attempt": attempts, "prompt": current_prompt, "sql": sql, "success": is_success, "error": error_msg})
        if is_success:
            success = True
            final_sql = sql
            break
        current_prompt = f"{prompt}\nPrevious attempt failed with error: {error_msg}\nPlease fix the SQL query."
        final_sql = sql
    return {"backend": backend_name, "model": model_name, "initial_prompt": prompt, "final_sql": final_sql, "success": success, "attempts": attempts, "history": history, "status": "completed"}


def run_agentic_loop(model_name: str, prompt: str | list[str], backend: str = "jax", context: AgentContext | None = None, **kwargs: JSONValue) -> JSONDict | list[JSONDict]:
    """Run an agentic self-correction loop.

        Args:
                    **kwargs: Advanced generation parameters (e.g., temperature, top_p, show_confidence).
    model_name: The name of the target model.
            prompt: The input text prompt.
            backend: The backend framework to use.
            context: The context.

        Returns:
            A list of results.
    """
    if context is None:
        context = AgentContext()
    db_kwargs = kwargs.get("db_kwargs")
    engine = LiveDatabaseEngine(db_path=context.db_path, ddl=context.ddl, db_type=context.db_type, db_kwargs=db_kwargs)
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    backend_impl = get_backend(backend)
    prompts = prompt if isinstance(prompt, list) else [prompt]

    async def _run_all() -> list[JSONDict]:
        """Execute logic.

        Returns:
            object: The resulting output from the operation.

        """
        tasks = [_process_single_prompt(backend, backend_impl, model_name, p, engine, context) for p in prompts]
        return await asyncio.gather(*tasks)

    try:
        results = asyncio.run(_run_all())
    finally:
        engine.close()
    return results if isinstance(prompt, list) else results[0]
