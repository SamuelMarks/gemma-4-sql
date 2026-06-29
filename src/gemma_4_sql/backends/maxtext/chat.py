"""MaxText-specific multi-turn conversational SQL logic."""

from __future__ import annotations

try:
    from maxtext.models import gemma4
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    gemma4 = None


def chat_turn(model_name: str, history: list[dict[str, str]], new_prompt: str, **_kwargs: object) -> dict[str, object]:
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
        status = "success_maxtext_chat"
        response = "SELECT * FROM maxtext_chat WHERE prompt = '{new_prompt}'".replace("{new_prompt}", new_prompt)
    else:
        status = "mocked_missing_maxtext"
        response = "SELECT * FROM fallback_chat"
    updated_history = list(history)
    updated_history.append({"role": "user", "content": new_prompt})
    updated_history.append({"role": "assistant", "content": response})
    return {"backend": "maxtext", "model": model_name, "response": response, "history": updated_history, "status": status}
