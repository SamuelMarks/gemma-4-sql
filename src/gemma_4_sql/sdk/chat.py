"""
SDK Chat module for Multi-Turn Conversational SQL.
"""

from __future__ import annotations

from typing import Any


def chat_turn(
    model_name: str,
    history: list[dict[str, str]],
    new_prompt: str,
    backend: str = "jax",
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Executes a single turn in a multi-turn SQL conversation.

    Args:
        model_name: The name of the model to use.
        history: The conversation history, as a list of dictionaries with 'role' and 'content'.
        new_prompt: The new user prompt.
        backend: The backend framework ('jax', 'keras', 'maxtext', 'pytorch').
        **kwargs: Additional parameters.

    Returns:
        A dictionary containing the response and the updated history.
    """
    from gemma_4_sql.sdk.registry import get_backend

    return get_backend(backend).chat_turn(model_name, history, new_prompt, **kwargs)
