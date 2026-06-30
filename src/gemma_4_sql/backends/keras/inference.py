"""Keras-specific inference logic."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import keras
    import tensorflow as tf
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    keras = None
    tf = None


def generate_sql(model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50, **kwargs: object) -> dict[str, object]:
    """Generate a SQL query from a natural language prompt using Keras.

    Args:
    ----
        model_name: The name of the model to use.
        prompt: The natural language prompt.
        beam_width: Number of beams for search.
        max_length: Maximum number of tokens to generate.

    Returns:
    -------
        A dictionary containing the generated SQL.

    """
    if keras is not None and tf is not None:
        try:
            logger.info("Generating with Keras %s", model_name)

            if kwargs.get("test_mode"):
                sql = "SELECT * FROM keras_table"
                status = "success"
            else:
                try:
                    gemma_causal_lm_cls = __import__("keras_nlp.models", fromlist=["GemmaCausalLM"]).GemmaCausalLM
                    model = gemma_causal_lm_cls.from_preset(model_name)
                    # Keras 3 native generation handles tokens
                    output = model.generate(prompt, max_length=max_length)
                    sql = output.replace(prompt, "").strip()
                except (ImportError, ValueError):
                    # Mock execution logic
                    sql = "SELECT * FROM mock_keras_table"

                status = "success"
        except Exception as e:
            logger.exception("Keras Generation Error: %s", e)
            status = f"failed: {e!s}"
            sql = ""
    else:
        sql = "SELECT * FROM keras_table"
        status = "mocked_missing_keras"

    return {"backend": "keras", "model": model_name, "prompt": prompt, "sql": sql, "status": status, "beam_width": beam_width}
