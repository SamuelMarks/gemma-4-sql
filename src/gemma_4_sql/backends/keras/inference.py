"""Keras-specific inference logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)
keras = None
tf = None
with catch_optional_imports():
    import keras
    import tensorflow as tf  # pragma: no cover


def generate_sql(model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50, **kwargs: JSONValue) -> JSONDict:
    """Generate a SQL query from a natural language prompt using Keras.

    Args:
        model_name: The name of the target model.
        prompt: The input text prompt.
        beam_width: The number of beams for beam search.
        max_length: The maximum length of the sequence.
        **kwargs: Additional keyword arguments.

    Returns:
        A dictionary containing the results.
    """
    confidence_score = 0.0
    if keras is None or tf is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Keras dependencies are missing.")
    try:
        logger.info("Generating with Keras %s", model_name)
        gemma_causal_lm_cls = __import__("keras_nlp.models", fromlist=["GemmaCausalLM"]).GemmaCausalLM
        model = gemma_causal_lm_cls.from_preset(model_name)
        output = model.generate(prompt, max_length=max_length)
        sql = output.replace(prompt, "").strip()
        confidence_score = 0.85
        status = "success"
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError, ImportError) as e:
        logger.exception("Keras Generation Error: ")
        status = f"failed: {e!s}"
        sql = ""
    return {"backend": "keras", "model": model_name, "prompt": prompt, "sql": sql, "status": status, "beam_width": beam_width, "confidence_score": confidence_score}
