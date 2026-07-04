# Copyright 2024
"""SDK Chat module for Multi-Turn Conversational SQL."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)


def chat_turn(model_name: str, history: list[dict[str, str]], new_prompt: str, backend: str = "jax", **kwargs: JSONValue) -> JSONDict:
    """Execute a single turn in a multi-turn SQL conversation.

    Args:
    ----
        model_name: The name of the model to use.
        history: The conversation history, as a list of dictionaries with 'role' and 'content'.
        new_prompt: The new user prompt.
        backend: The backend framework ('jax', 'keras', 'maxtext', 'pytorch').
        **kwargs: Additional parameters.

    Returns:
    -------
        A dictionary containing the response and the updated history.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    backend_impl = get_backend(backend)
    try:
        full_prompt = ""
        for turn in history:
            full_prompt += f"{turn['role']}: {turn['content']}\n"
        full_prompt += f"user: {new_prompt}\nassistant: "
        result = backend_impl.generate_sql(model_name, full_prompt, **kwargs)
        response = str(result.get("sql", "SELECT * FROM fallback"))
        status = f"success_{backend}_chat"
    except (RuntimeError, ValueError, KeyError) as e:
        logger.exception("%s chat error", backend.capitalize())
        status = f"failed: {e!s}"
        response = "SELECT * FROM fallback_chat"
    updated_history = list(history)
    updated_history.extend(({"role": "user", "content": new_prompt}, {"role": "assistant", "content": response}))
    return {"backend": backend, "model": model_name, "response": response, "history": updated_history, "status": status}
