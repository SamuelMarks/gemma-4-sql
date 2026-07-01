"""JAX-specific multi-turn conversational SQL logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.backends.jax.inference import generate_sql

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue

try:
    import jax
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    jax = None


def chat_turn(model_name: str, history: list[dict[str, str]], new_prompt: str, **kwargs: JSONValue) -> JSONDict:
    """Execute a single turn in a multi-turn SQL conversation using JAX.

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
    if jax is not None:
        # Build prompt from history
        full_prompt = ""
        for turn in history:
            full_prompt += f"{turn['role']}: {turn['content']}\n"
        full_prompt += f"user: {new_prompt}\nassistant: "

        # Use actual generation logic
        result = generate_sql(model_name, full_prompt, **kwargs)
        response = str(result.get("sql", "SELECT * FROM fallback"))
        status = "success_jax_chat"
    else:
        status = "mocked_missing_jax"
        response = "SELECT * FROM fallback_chat"

    updated_history = list(history)
    updated_history.append({"role": "user", "content": new_prompt})
    updated_history.append({"role": "assistant", "content": response})
    return {"backend": "jax", "model": model_name, "response": response, "history": updated_history, "status": status}
