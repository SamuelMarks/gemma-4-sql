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
    from gemma_4_sql.sdk.registry import get_backend

    return get_backend(backend).generate_sql(model_name, prompt, beam_width, max_length)
