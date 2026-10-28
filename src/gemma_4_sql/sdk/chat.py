"""SDK Chat module for Multi-Turn Conversational SQL."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from gemma_4_sql.constants import DEFAULT_BACKEND
from gemma_4_sql.exceptions import ChatError, InferenceError, is_failed_status
from gemma_4_sql.sdk.agent import extract_sql_from_response
from gemma_4_sql.sdk.registry import get_backend
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
        ChatError: If backend does not return SQL or chat turn execution fails.

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
            raise ChatError(f"Backend {backend} did not return SQL.", backend=backend, model_name=model_name)
        response = extract_sql_from_response(str(result["sql"]))
        status = f"success_{backend}_chat"
    except (ChatError, InferenceError, RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("%s chat error", backend.capitalize())
        status = f"failed: {e!s}"
        raise ChatError(f"Chat turn failed: {e!s}", backend=backend, model_name=model_name, cause=e) from e
    updated_history: list[ChatMessageDict] = [{"role": str(turn["role"]), "content": str(turn["content"])} for turn in history]
    user_msg: ChatMessageDict = {"role": "user", "content": new_prompt}
    assistant_msg: ChatMessageDict = {"role": "assistant", "content": response}
    updated_history.extend((user_msg, assistant_msg))
    return {
        "backend": backend,
        "model": model_name,
        "response": response,
        "history": updated_history,
        "status": status,
    }


__all__ = ["chat_turn"]
