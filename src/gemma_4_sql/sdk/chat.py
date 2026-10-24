"""SDK Chat module for Multi-Turn Conversational SQL."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.constants import DEFAULT_BACKEND
from gemma_4_sql.exceptions import InferenceError, is_failed_status
from gemma_4_sql.sdk.registry import get_backend

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from gemma_4_sql.type_hints import ChatMessageDict, ChatTurnResultDict, JSONValue

logger = logging.getLogger(__name__)


def chat_turn(
    model_name: str,
    history: Sequence[Mapping[str, str] | ChatMessageDict],
    new_prompt: str,
    backend: str = DEFAULT_BACKEND,
    **kwargs: JSONValue,
) -> ChatTurnResultDict:
    """Execute a single turn in a multi-turn SQL conversation.

    Args:
        model_name: The name of the target model.
        history: A sequence of prior conversation turns with `'role'` and `'content'` keys.
        new_prompt: The string representing the new user prompt.
        backend: The backend framework to use.
        **kwargs: Advanced generation parameters (e.g., temperature, top_p, show_confidence).

    Returns:
        A structured `ChatTurnResultDict` containing the response and updated history.

    Raises:
        ValueError: If backend does not return SQL.
        InferenceError: If chat turn execution fails.

    """
    backend_impl = get_backend(backend)
    try:
        lines = [f"{turn['role']}: {turn['content']}" for turn in history]
        lines.extend((f"user: {new_prompt}", "assistant: "))
        full_prompt = chr(10).join(lines)
        call_kwargs = dict(kwargs)
        bw_raw = call_kwargs.pop("beam_width", 3)
        ml_raw = call_kwargs.pop("max_length", 50)
        bw = int(bw_raw) if isinstance(bw_raw, (int, float, str)) else 3
        ml = int(ml_raw) if isinstance(ml_raw, (int, float, str)) else 50
        result = backend_impl.generate_sql(model_name, full_prompt, beam_width=bw, max_length=ml, **call_kwargs)
        if "sql" not in result or is_failed_status(result):
            raise InferenceError(f"Backend {backend} did not return SQL.", backend=backend, model_name=model_name)
        response = str(result["sql"])
        status = f"success_{backend}_chat"
    except (InferenceError, RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("%s chat error", backend.capitalize())
        status = f"failed: {e!s}"
        raise InferenceError(f"Chat turn failed: {e!s}", backend=backend, model_name=model_name, cause=e) from e
    updated_history: list[ChatMessageDict] = [{"role": str(turn["role"]), "content": str(turn["content"])} for turn in history]
    updated_history.extend(({"role": "user", "content": new_prompt}, {"role": "assistant", "content": response}))
    return {
        "backend": backend,
        "model": model_name,
        "response": response,
        "history": updated_history,
        "status": status,
    }


__all__ = ["chat_turn"]
