"""SDK Chat module for Multi-Turn Conversational SQL using Gemma 4 turn markers."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from gemma_4_sql.constants import DEFAULT_BACKEND
from gemma_4_sql.exceptions import ChatError, ConfigurationError, InferenceError, is_failed_status
from gemma_4_sql.sdk.agent import extract_sql_from_response
from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.type_hints import ChatMessageDict, ChatTurnResultDict, JSONValue

logger = logging.getLogger(__name__)

VALID_CHAT_ROLES: frozenset[str] = frozenset({"user", "assistant", "model", "system"})
_TURN_START = "<start_of_turn>"
_TURN_END = "<end_of_turn>"
_NL = """
"""


def _validate_and_normalize_turn(turn: Mapping[str, JSONValue] | ChatMessageDict) -> tuple[str, str]:
    """Validate a conversation history turn and guard against turn-marker spoofing.

    Args:
        turn: Conversation turn mapping containing `'role'` and `'content'`.

    Returns:
        Tuple of `(normalized_role, validated_content)`.

    Raises:
        ConfigurationError: If `'role'` or `'content'` is missing, non-string, invalid,
            or contains embedded Gemma 4 turn control tokens.

    """
    if "role" not in turn or "content" not in turn:
        msg = "Each chat history entry must contain both 'role' and 'content' keys."
        raise ConfigurationError(msg)
    raw_role = turn["role"]
    raw_content = turn["content"]
    if not isinstance(raw_role, str) or not isinstance(raw_content, str):
        msg = "Chat history 'role' and 'content' values must be strings."
        raise ConfigurationError(msg)
    role = raw_role.strip().lower()
    if role not in VALID_CHAT_ROLES:
        msg = f"Unsupported chat role {raw_role!r}; expected one of {sorted(VALID_CHAT_ROLES)}."
        raise ConfigurationError(msg)
    if _TURN_START in raw_content or _TURN_END in raw_content:
        msg = "Embedded turn control tokens (<start_of_turn>/<end_of_turn>) are disallowed in chat content."
        raise ConfigurationError(msg)
    return (role, raw_content)


def format_gemma4_chat_prompt(validated_turns: Sequence[tuple[str, str]], new_prompt: str) -> str:
    """Format multi-turn conversation history using Gemma 4 `<start_of_turn>` / `<end_of_turn>` markers.

    Args:
        validated_turns: Sequence of `(role, content)` pairs.
        new_prompt: Validated current user turn text.

    Returns:
        Formatted Gemma 4 multi-turn prompt string ending at the model turn prefix.

    """
    blocks: list[str] = []
    for role, content in validated_turns:
        gemma_role = "model" if role in {"assistant", "model"} else role
        blocks.append(f"{_TURN_START}{gemma_role}{_NL}{content}{_TURN_END}")
    blocks.extend((f"{_TURN_START}user{_NL}{new_prompt}{_TURN_END}", f"{_TURN_START}model{_NL}"))
    return _NL.join(blocks)


def chat_turn(
    model_name: str,
    history: Sequence[Mapping[str, JSONValue] | ChatMessageDict],
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
        **kwargs: Advanced generation parameters (e.g., `temperature`, `top_p`, `show_confidence`).

    Returns:
        A structured `ChatTurnResultDict` containing the response and updated history.

    Raises:
        ConfigurationError: If `backend`, `history`, or `new_prompt` fails validation.
        ChatError: If backend does not return SQL or chat turn execution fails.

    """
    if _TURN_START in new_prompt or _TURN_END in new_prompt:
        msg = "Embedded turn control tokens (<start_of_turn>/<end_of_turn>) are disallowed in new_prompt."
        raise ConfigurationError(msg, backend=backend, model_name=model_name)
    validated_turns = [_validate_and_normalize_turn(turn) for turn in history]
    backend_impl = get_backend(backend)
    full_prompt = format_gemma4_chat_prompt(validated_turns, new_prompt)

    try:
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
    except ChatError:
        raise
    except (InferenceError, RuntimeError, ValueError, OSError) as e:
        logger.exception("%s chat error", backend.capitalize())
        raise ChatError(f"Chat turn failed: {e!s}", backend=backend, model_name=model_name, cause=e) from e

    updated_history: list[ChatMessageDict] = [{"role": role, "content": content} for role, content in validated_turns]
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


__all__ = ["VALID_CHAT_ROLES", "chat_turn", "format_gemma4_chat_prompt"]
