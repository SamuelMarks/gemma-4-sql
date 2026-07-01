"""MaxText-specific multi-turn conversational SQL logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.maxtext.inference import generate_sql

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue

logger = logging.getLogger(__name__)

try:
    from maxtext.models import gemma4
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    gemma4 = None


def chat_turn(model_name: str, history: list[dict[str, str]], new_prompt: str, **kwargs: JSONValue) -> JSONDict:
    """Execute a single turn in a multi-turn SQL conversation using MaxText.

    Args:
    ----
        model_name: The name of the model.
        history: Previous conversation history.
        new_prompt: The new user prompt.
        **kwargs: Additional parameters.

    Returns:
    -------
        A dictionary containing the response and updated history.

    """
    if gemma4 is not None:
        try:
            full_prompt = ""
            for turn in history:
                full_prompt += f"{turn['role']}: {turn['content']}\n"
            full_prompt += f"user: {new_prompt}\nassistant: "

            result = generate_sql(model_name, full_prompt, **kwargs)
            response = str(result.get("sql", ""))
            status = "success_maxtext_chat"
        except Exception as e:
            logger.exception("MaxText chat error: %s", e)
            status = f"failed: {e!s}"
            response = "SELECT * FROM fallback_chat"
    else:
        status = "mocked_missing_maxtext"
        response = "SELECT * FROM fallback_chat"

    updated_history = list(history)
    updated_history.append({"role": "user", "content": new_prompt})
    updated_history.append({"role": "assistant", "content": response})
    return {"backend": "maxtext", "model": model_name, "response": response, "history": updated_history, "status": status}
