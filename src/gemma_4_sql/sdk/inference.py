# Copyright 2024
"""SDK Inference module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict


def generate(model_name: str, prompt: str, backend: str = "jax", beam_width: int = 3, max_length: int = 50, **kwargs: object) -> JSONDict:
    """Generate a SQL query from a natural language prompt using Beam Search.

    Args:
    ----
        model_name: The name or path of the model.
        prompt: The natural language prompt.
        backend: The backend framework ('jax', 'keras', 'maxtext', or 'pytorch').
        beam_width: The number of beams to maintain during search.
        max_length: The maximum generation length.
        **kwargs: Additional parameters like `show_confidence`.

    Returns:
    -------
        Generation results dictionary containing the output SQL and status.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    result = get_backend(backend).generate_sql(model_name, prompt, beam_width, max_length)
    kwargs.get("show_confidence") and "confidence_score" in result
    return result
