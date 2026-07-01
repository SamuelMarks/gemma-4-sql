"""MLX-specific multi-turn conversational SQL logic."""

from __future__ import annotations

import logging

from gemma_4_sql.backends.mlx.inference import generate_sql

logger = logging.getLogger(__name__)

try:
    import mlx
    from transformers import AutoTokenizer
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    mlx = None
    AutoTokenizer = None


def chat_turn(model_name: str, history: list[dict[str, str]], new_prompt: str, **kwargs: object) -> dict[str, object]:
    """Execute a single turn in a multi-turn SQL conversation using MLX.

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
    if mlx is not None and AutoTokenizer is not None:
        try:
            updated_history = list(history)
            updated_history.append({"role": "user", "content": new_prompt})

            if kwargs.get("test_mode"):
                full_prompt = new_prompt
                response = "SELECT * FROM generated_chat"
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                full_prompt = tokenizer.apply_chat_template(updated_history, tokenize=False, add_generation_prompt=True)

                result = generate_sql(model_name, full_prompt, **kwargs)
                response = str(result.get("sql", ""))

            status = "success_mlx_chat"
        except Exception as e:
            logger.exception("Chat failed: %s", e)
            status = f"failed: {e!s}"
            response = "SELECT * FROM fallback_chat"
            updated_history = list(history)
            updated_history.append({"role": "user", "content": new_prompt})
    else:
        status = "mocked_missing_mlx"
        response = "SELECT * FROM fallback_chat"
        updated_history = list(history)
        updated_history.append({"role": "user", "content": new_prompt})

    updated_history.append({"role": "assistant", "content": response})
    return {"backend": "mlx", "model": model_name, "response": response, "history": updated_history, "status": status}
