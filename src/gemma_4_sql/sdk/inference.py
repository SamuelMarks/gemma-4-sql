"""
SDK Inference module.
"""

from __future__ import annotations

from typing import Any

def generate(
    model_name: str,
    prompt: str,
    backend: str = "jax",
    beam_width: int = 3,
    max_length: int = 50,
) -> dict[str, Any]:
    """
    Generates a SQL query from a natural language prompt using Beam Search.

    Args:
        model_name: The name or path of the model.
        prompt: The natural language prompt.
        backend: The backend framework ('jax', 'keras', 'maxtext', or 'pytorch').
        beam_width: The number of beams to maintain during search.
        max_length: The maximum generation length.

    Returns:
        Generation results dictionary containing the output SQL and status.
    """
    if backend == "jax":
        from gemma_4_sql.backends.jax.inference import generate_sql as jax_gen

        return jax_gen(model_name, prompt, beam_width, max_length)
    elif backend == "keras":
        from gemma_4_sql.backends.keras.inference import (
            generate_sql as keras_gen,
        )

        return keras_gen(model_name, prompt, beam_width, max_length)
    elif backend == "maxtext":
        from gemma_4_sql.backends.maxtext.inference import generate_sql as maxtext_gen

        return maxtext_gen(model_name, prompt, beam_width, max_length)
    elif backend == "pytorch":
        from gemma_4_sql.backends.pytorch.inference import (
            generate_sql as pytorch_gen,
        )

        return pytorch_gen(model_name, prompt, beam_width, max_length)
    else:
        raise ValueError(f"Unknown backend: {backend}")
