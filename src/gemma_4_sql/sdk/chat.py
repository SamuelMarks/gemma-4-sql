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
        model_name: The name of the target model.
        history: A sequence of history.
        new_prompt: The string representing the new prompt.
        backend: The backend framework to use.
        **kwargs: Additional keyword arguments.

    Returns:
        A dictionary containing the results.
    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    backend_impl = get_backend(backend)
    try:
        full_prompt = ""
        for turn in history:
            full_prompt += f"{turn['role']}: {turn['content']}\n"
        full_prompt += f"user: {new_prompt}\nassistant: "
        result = backend_impl.generate_sql(model_name, full_prompt, **kwargs)
        if "sql" not in result:
            raise ValueError(f"Backend {backend} did not return SQL.")
        response = str(result["sql"])
        status = f"success_{backend}_chat"
    except Exception as e:
        logger.exception("%s chat error", backend.capitalize())
        status = f"failed: {e!s}"
        raise RuntimeError(f"Chat turn failed: {e!s}") from e
    updated_history = list(history)
    updated_history.extend(({"role": "user", "content": new_prompt}, {"role": "assistant", "content": response}))
    return {"backend": backend, "model": model_name, "response": response, "history": updated_history, "status": status}
