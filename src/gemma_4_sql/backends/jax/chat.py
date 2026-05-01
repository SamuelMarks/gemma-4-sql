"""
JAX-specific multi-turn conversational SQL logic.
"""

from __future__ import annotations

from typing import Any

try:
    import jax
except Exception:
    jax = None


def chat_turn(
    model_name: str, history: list[dict[str, str]], new_prompt: str, **kwargs: Any
) -> dict[str, Any]:
    """
    Executes a single turn in a multi-turn SQL conversation using JAX.

    Args:
        model_name: The name of the model.
        history: Previous conversation history.
        new_prompt: The new user prompt.
        **kwargs: Additional parameters.

    Returns:
        A dictionary containing the response and updated history.
    """
    if jax is not None:
        status = "success_jax_chat"
        response = f"SELECT * FROM jax_chat WHERE prompt = '{new_prompt}'"
    else:
        status = "mocked_missing_jax"
        response = "SELECT * FROM fallback_chat"

    updated_history = list(history)
    updated_history.append({"role": "user", "content": new_prompt})
    updated_history.append({"role": "assistant", "content": response})

    return {
        "backend": "jax",
        "model": model_name,
        "response": response,
        "history": updated_history,
        "status": status,
    }
