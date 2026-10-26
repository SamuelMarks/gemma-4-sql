"""SDK Agent module for self-correction execution feedback loops."""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from gemma_4_sql.constants import DEFAULT_BACKEND
from gemma_4_sql.exceptions import is_failed_status
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
from gemma_4_sql.sdk.protocols import BackendProtocol
from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.type_hints import AgentAttemptDict, AgentLoopResultDict, JSONPrimitive, JSONValue

MAX_ERR_LEN = 500


@dataclass
class AgentContext:
    """Context for the agentic self-correction loop.

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
    image_path: str | Path | None = None
    audio_path: str | Path | None = None
    modality: str = "text"
    multimodal_context: dict[str, Any] | None = None


def extract_sql_from_response(raw_sql: str) -> str:
    """Extract clean SQL query string from a raw model response or Markdown code fence.

    Args:
        raw_sql: Raw string output from the model.

    Returns:
        Cleaned SQL query string.

    """
    text = raw_sql.strip()
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


_extract_sql_from_response = extract_sql_from_response


def classify_error_hint(error_msg: str | None) -> str:
    """Classify a database execution error message into an actionable diagnostic hint.

    Args:
        error_msg: Database execution error string or None.

    Returns:
        Diagnostic hint suffix string (or empty string if no specific hint applies).

    """
    if not error_msg:
        return ""
    if re.search(r"(?:no such column|column .* does not exist)", error_msg, re.IGNORECASE):
        return " [Hint: check valid column names]"
    if re.search(r"(?:no such table|relation .* does not exist)", error_msg, re.IGNORECASE):
        return " [Hint: check available table names]"
    return ""


_classify_error_hint = classify_error_hint


def _prepare_attempt_kwargs(
    context: AgentContext,
    attempts: int,
    gen_kwargs: dict[str, JSONValue],
) -> tuple[int, int, dict[str, Any]]:
    """Build generation parameters for the current self-correction attempt.

    Args:
        context: AgentContext containing multimodal and confidence settings.
        attempts: 1-based attempt counter.
        gen_kwargs: Base generation keyword arguments.

    Returns:
        Tuple of `(beam_width, max_length, extra_generation_kwargs)`.

    """
    b_width = int(str(gen_kwargs.get("beam_width", 3)))
    m_len = int(str(gen_kwargs.get("max_length", 50)))
    base_temp = float(str(gen_kwargs.get("temperature", 0.0)))
    current_temp = min(1.0, base_temp + (attempts - 1) * 0.1)

    other_kwargs: dict[str, Any] = {k: v for k, v in gen_kwargs.items() if k not in ("beam_width", "max_length", "temperature")}
    other_kwargs["temperature"] = current_temp
    if context.image_path is not None:
        other_kwargs["image_path"] = str(context.image_path)
    if context.audio_path is not None:
        other_kwargs["audio_path"] = str(context.audio_path)
    other_kwargs["modality"] = context.modality
    if context.multimodal_context is not None:
        other_kwargs["multimodal_context"] = context.multimodal_context
    return (b_width, m_len, other_kwargs)


def _format_base_prompt(prompt: str, context: AgentContext) -> str:
    """Format prompt with multimodal placeholders when image or audio paths are present.

    Args:
        prompt: Raw natural language query string.
        context: AgentContext with optional image/audio paths.

    Returns:
        Formatted base prompt string.

    """
    if context.image_path or context.audio_path:
        from gemma_4_sql.backends.common_multimodal import format_multimodal_prompt

        mm = format_multimodal_prompt(
            prompt,
            has_image=context.image_path is not None,
            has_audio=context.audio_path is not None,
        )
        return str(mm.get("prompt", prompt))
    return prompt


async def _process_single_prompt(
    backend_name: str,
    backend_impl: BackendProtocol,
    model_name: str,
    prompt: str,
    engine: LiveDatabaseEngine,
    context: AgentContext,
    **gen_kwargs: JSONValue,
) -> AgentLoopResultDict:
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
        Result dictionary containing `final_sql`, `history`, and `status`.

    """
    base_prompt = _format_base_prompt(prompt, context)
    current_prompt = base_prompt
    attempts = 0
    success = False
    final_sql = ""
    final_results: list[tuple[JSONPrimitive, ...]] = []
    history: list[AgentAttemptDict] = []
    while attempts < context.max_retries:
        attempts += 1
        (b_width, m_len, other_kwargs) = _prepare_attempt_kwargs(context, attempts, gen_kwargs)
        gen_res = backend_impl.generate_sql(model_name, current_prompt, beam_width=b_width, max_length=m_len, **other_kwargs)
        sql = _extract_sql_from_response(str(gen_res.get("sql", ""))) if not is_failed_status(gen_res) else ""
        confidence_score = float(str(gen_res.get("confidence_score", 1.0)))
        if context.min_confidence and confidence_score < context.min_confidence:
            history.append({
                "attempt": attempts,
                "prompt": current_prompt,
                "sql": sql,
                "success": False,
                "error": f"Confidence score {confidence_score:.2f} below threshold {context.min_confidence}",
            })
            current_prompt = f"""{base_prompt}
Previous attempt was rejected due to low confidence ({confidence_score:.2f}). Please provide a more certain SQL query."""
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

        hint = _classify_error_hint(error_msg)
        current_prompt = f"""{base_prompt}
Previous attempt failed with error: {error_msg}{hint}
Please fix the SQL query."""
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


@overload
async def run_agentic_loop_async(
    model_name: str,
    prompt: str,
    backend: str = DEFAULT_BACKEND,
    context: AgentContext | None = None,
    **kwargs: JSONValue,
) -> AgentLoopResultDict:
    """Asynchronously run an agentic self-correction loop for a single prompt."""


@overload
async def run_agentic_loop_async(
    model_name: str,
    prompt: list[str],
    backend: str = DEFAULT_BACKEND,
    context: AgentContext | None = None,
    **kwargs: JSONValue,
) -> list[AgentLoopResultDict]:
    """Asynchronously run an agentic self-correction loop for a batch of prompts."""


@overload
async def run_agentic_loop_async(
    model_name: str,
    prompt: str | list[str],
    backend: str = DEFAULT_BACKEND,
    context: AgentContext | None = None,
    **kwargs: JSONValue,
) -> AgentLoopResultDict | list[AgentLoopResultDict]:
    """Asynchronously run an agentic self-correction loop for a prompt or prompt list."""


async def run_agentic_loop_async(
    model_name: str,
    prompt: str | list[str],
    backend: str = DEFAULT_BACKEND,
    context: AgentContext | None = None,
    **kwargs: JSONValue,
) -> AgentLoopResultDict | list[AgentLoopResultDict]:
    """Asynchronously run an agentic self-correction loop.

    Args:
        model_name: The name of the target model.
        prompt: Single natural language prompt or list of prompts.
        backend: The backend framework to use.
        context: AgentContext for state tracking across execution iterations.
        **kwargs: Advanced generation parameters (e.g., `temperature`, `top_p`, `db_kwargs`).

    Returns:
        A single result dictionary or list of result dictionaries matching `prompt`.

    """
    if context is None:
        context = AgentContext()
    db_kwargs = kwargs.get("db_kwargs")
    gen_kwargs = {k: v for k, v in kwargs.items() if k != "db_kwargs"}
    engine = LiveDatabaseEngine(db_path=context.db_path, ddl=context.ddl, db_type=context.db_type, db_kwargs=db_kwargs)
    backend_impl = get_backend(backend)
    prompts = prompt if isinstance(prompt, list) else [prompt]
    try:
        tasks = [_process_single_prompt(backend, backend_impl, model_name, p, engine, context, **gen_kwargs) for p in prompts]
        results = list(await asyncio.gather(*tasks))
    finally:
        await engine.close_async()
    return results if isinstance(prompt, list) else results[0]


@overload
def run_agentic_loop(
    model_name: str,
    prompt: str,
    backend: str = DEFAULT_BACKEND,
    context: AgentContext | None = None,
    **kwargs: JSONValue,
) -> AgentLoopResultDict:
    """Run an agentic self-correction loop synchronously for a single prompt."""


@overload
def run_agentic_loop(
    model_name: str,
    prompt: list[str],
    backend: str = DEFAULT_BACKEND,
    context: AgentContext | None = None,
    **kwargs: JSONValue,
) -> list[AgentLoopResultDict]:
    """Run an agentic self-correction loop synchronously for a batch of prompts."""


@overload
def run_agentic_loop(
    model_name: str,
    prompt: str | list[str],
    backend: str = DEFAULT_BACKEND,
    context: AgentContext | None = None,
    **kwargs: JSONValue,
) -> AgentLoopResultDict | list[AgentLoopResultDict]:
    """Run an agentic self-correction loop synchronously for a prompt or prompt list."""


def run_agentic_loop(
    model_name: str,
    prompt: str | list[str],
    backend: str = DEFAULT_BACKEND,
    context: AgentContext | None = None,
    **kwargs: JSONValue,
) -> AgentLoopResultDict | list[AgentLoopResultDict]:
    """Run an agentic self-correction loop synchronously.

    Args:
        model_name: The name of the target model.
        prompt: The input text prompt or list of prompts.
        backend: The backend framework to use.
        context: The agent context for state tracking across execution iterations.
        **kwargs: Advanced generation parameters (e.g., `temperature`, `top_p`, `show_confidence`).

    Returns:
        A single result dictionary or list of result dictionaries.

    """
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None and running_loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(
                asyncio.run,
                run_agentic_loop_async(model_name, prompt, backend=backend, context=context, **kwargs),
            ).result()
    return asyncio.run(run_agentic_loop_async(model_name, prompt, backend=backend, context=context, **kwargs))


__all__ = [
    "MAX_ERR_LEN",
    "AgentContext",
    "_classify_error_hint",
    "_extract_sql_from_response",
    "classify_error_hint",
    "extract_sql_from_response",
    "run_agentic_loop",
    "run_agentic_loop_async",
]
