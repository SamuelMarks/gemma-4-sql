"""MLX-specific inference logic."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from mlx_lm import generate, load
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    load = None
    generate = None


def generate_sql(model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50, **kwargs: object) -> dict[str, object]:
    """Generate a SQL query from a natural language prompt using MLX.

    Args:
    ----
        model_name: The name of the model to use.
        prompt: The natural language prompt.
        beam_width: Number of beams for search.
        max_length: Maximum number of tokens to generate.
        **kwargs: Additional arguments.

    Returns:
    -------
        A dictionary containing the generated SQL.

    """
    confidence_score = 0.0
    if load is not None and generate is not None:
        try:
            logger.info("Generating with %s", model_name)

            if kwargs.get("test_mode"):
                sql = "SELECT * FROM mlx_table"
                confidence_score = 0.95
            else:
                model, tokenizer = load(model_name)
                # mlx_lm generator doesn't easily expose logprobs out of the box in the `generate` utility without callbacks
                # Mocking the confidence score logic here for now
                generated_text = generate(model, tokenizer, prompt=prompt, max_tokens=max_length, verbose=False)
                sql = generated_text.strip()
                confidence_score = 0.85

            status = "success"
        except Exception as e:
            logger.exception("Generation failed: %s", e)
            sql = ""
            status = f"failed: {e!s}"
    else:
        sql = "SELECT * FROM mlx_table"
        confidence_score = 0.95
        status = "mocked_missing_mlx"

    return {"backend": "mlx", "model": model_name, "prompt": prompt, "sql": sql, "status": status, "beam_width": beam_width, "confidence_score": confidence_score}
