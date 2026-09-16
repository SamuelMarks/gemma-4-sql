"""SDK Agent module for self-correction execution feedback loops."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine

MAX_ERR_LEN = 500
if TYPE_CHECKING:
    from gemma_4_sql.sdk.protocols import BackendProtocol
    from gemma_4_sql.type_hints import JSONDict, JSONPrimitive, JSONValue


@dataclass
class AgentContext:
    """Context for the agentic loop.

    Attributes:
        db_path: Path to live or in-memory SQLite/DuckDB database.
        ddl: Schema DDL definitions.
        db_type: Target database dialect.
        max_retries: Maximum self-correction attempts.
        min_confidence: Threshold below which generated SQL is rejected.
        image_path: Optional path to schema diagram or ERD image.
        audio_path: Optional path to recorded natural language query audio.
        modality: Explicit modality selector ('text', 'vision', 'audio', 'multimodal').
        multimodal_context: Optional cached multimodal embeddings or metadata across attempts.
    """

    db_path: str = ":memory:"
    ddl: str | None = None
    db_type: str = "sqlite"
    max_retries: int = 3
    min_confidence: float = 0.0
    image_path: str | Any | None = None
    audio_path: str | Any | None = None
    modality: str = "text"
    multimodal_context: dict[str, Any] | None = None


async def _process_single_prompt(
    backend_name: str,
    backend_impl: BackendProtocol,
    model_name: str,
    prompt: str,
    engine: LiveDatabaseEngine,
    context: AgentContext,
    **gen_kwargs: JSONValue,
) -> JSONDict:
    """Execute self-correction loop preserving multimodal contextual embeddings across turns.

    Args:
        backend_name: Name of backend engine.
        backend_impl: Protocol backend implementation.
        model_name: Target model identifier.
        prompt: Natural language query prompt.
        engine: LiveDatabaseEngine instance.
        context: AgentContext containing retry budget and multimodal settings.
        **gen_kwargs: Generation options.

    Returns:
        Result dictionary containing final_sql, history, and status.
    """
    if context.image_path or context.audio_path:
        from gemma_4_sql.backends.common_multimodal import format_multimodal_prompt

        mm = format_multimodal_prompt(
            prompt,
            has_image=context.image_path is not None,
            has_audio=context.audio_path is not None,
        )
        base_prompt = mm["prompt"]
    else:
        base_prompt = prompt

    current_prompt = base_prompt
    attempts = 0
    success = False
    final_sql = ""
    final_results: list[tuple[JSONPrimitive, ...]] = []
    history: list[JSONDict] = []
    while attempts < context.max_retries:
        attempts += 1
        b_width = int(str(gen_kwargs.get("beam_width", 3)))
        m_len = int(str(gen_kwargs.get("max_length", 50)))
        base_temp = float(str(gen_kwargs.get("temperature", 0.0)))
        current_temp = min(1.0, base_temp + (attempts - 1) * 0.1)

        other_kwargs = {k: v for k, v in gen_kwargs.items() if k not in ("beam_width", "max_length", "temperature")}
        other_kwargs["temperature"] = current_temp
        if context.image_path is not None:
            other_kwargs["image_path"] = str(context.image_path)
        if context.audio_path is not None:
            other_kwargs["audio_path"] = str(context.audio_path)
        other_kwargs["modality"] = context.modality
        if context.multimodal_context is not None:
            other_kwargs["multimodal_context"] = context.multimodal_context

        try:
            gen_res = backend_impl.generate_sql(model_name, current_prompt, beam_width=b_width, max_length=m_len, **other_kwargs)
        except TypeError:
            gen_res = backend_impl.generate_sql(model_name, current_prompt)
        sql = str(gen_res.get("sql", ""))
        confidence_score = float(str(gen_res.get("confidence_score", 1.0)))
        if context.min_confidence and confidence_score < context.min_confidence:
            history.append({"attempt": attempts, "prompt": current_prompt, "sql": sql, "success": False, "error": f"Confidence score {confidence_score:.2f} below threshold {context.min_confidence}"})
            current_prompt = f"{base_prompt}\nPrevious attempt was rejected due to low confidence ({confidence_score:.2f}). Please provide a more certain SQL query."
            final_sql = sql
            continue
        (is_success, query_results, error_msg) = await engine.execute_with_feedback_async(sql)
        error_msg = error_msg[:MAX_ERR_LEN] + "... (truncated)" if error_msg and len(error_msg) > MAX_ERR_LEN else error_msg
        history.append({"attempt": attempts, "prompt": current_prompt, "sql": sql, "success": is_success, "error": error_msg})
        if is_success:
            success = True
            final_sql = sql
            final_results = query_results
            break

        hint = ""
        if error_msg:
            if re.search(r"(?:no such column|column .* does not exist)", error_msg, re.IGNORECASE):
                hint = " [Hint: check valid column names]"
            if not hint and re.search(r"(?:no such table|relation .* does not exist)", error_msg, re.IGNORECASE):
                hint = " [Hint: check available table names]"

        current_prompt = f"{base_prompt}\nPrevious attempt failed with error: {error_msg}{hint}\nPlease fix the SQL query."
        final_sql = sql
    return {
        "backend": backend_name,
        "model": model_name,
        "initial_prompt": prompt,
        "final_sql": final_sql,
        "results": final_results,
        "success": success,
        "attempts": attempts,
        "history": history,
        "status": "completed",
    }


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
    gen_kwargs = {k: v for k, v in kwargs.items() if k != "db_kwargs"}
    engine = LiveDatabaseEngine(db_path=context.db_path, ddl=context.ddl, db_type=context.db_type, db_kwargs=db_kwargs)
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    backend_impl = get_backend(backend)
    prompts = prompt if isinstance(prompt, list) else [prompt]

    async def _run_all() -> list[JSONDict]:
        """Execute logic.

        Returns:
            object: The resulting output from the operation.

        """
        tasks = [_process_single_prompt(backend, backend_impl, model_name, p, engine, context, **gen_kwargs) for p in prompts]
        return await asyncio.gather(*tasks)

    try:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None and running_loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                results = executor.submit(asyncio.run, _run_all()).result()
        else:
            results = asyncio.run(_run_all())
    finally:
        engine.close()
    return results if isinstance(prompt, list) else results[0]
