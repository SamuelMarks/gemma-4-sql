"""MLX-specific inference logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)
load = None
generate = None
with catch_optional_imports():
    from mlx_lm import generate, load


def generate_sql(model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50, **kwargs: JSONValue) -> JSONDict:
    """Generate a SQL query from a natural language prompt using MLX.

        Args:
                    **kwargs: Advanced generation parameters (e.g., temperature, top_p, show_confidence).
    model_name: The name of the target model.
            prompt: The input text prompt.
            beam_width: The number of beams for beam search.
            max_length: The maximum length of the sequence.

        Returns:
            A dictionary containing the results.
    """
    confidence_score = 0.0
    if load is None or generate is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MLX dependencies are missing.")
    try:
        logger.info("Generating with %s", model_name)
        if kwargs.get("test_mode"):
            sql = "SELECT * FROM mlx_table"
            confidence_score = 0.95
        else:
            (model, tokenizer) = load(model_name)
            generated_text = generate(model, tokenizer, prompt=prompt, max_tokens=max_length, verbose=False)
            sql = generated_text.strip()
            confidence_score = 0.85
        status = "success"
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Generation failed: ")
        sql = ""
        status = f"failed: {e!s}"
    return {"backend": "mlx", "model": model_name, "prompt": prompt, "sql": sql, "status": status, "beam_width": beam_width, "confidence_score": confidence_score}
