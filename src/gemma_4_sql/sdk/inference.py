"""SDK Inference module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict


def generate(model_name: str, prompt: str, backend: str = "jax", beam_width: int = 3, max_length: int = 50, **kwargs: object) -> JSONDict:
    """Generate a SQL query from a natural language prompt using Beam Search.

    Args:
        model_name: The name of the target model.
        prompt: The input text prompt.
        backend: The backend framework to use.
        beam_width: The number of beams for beam search.
        max_length: The maximum length of the sequence.
        **kwargs: Additional keyword arguments.

    Returns:
        A dictionary containing the results.
    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    result = get_backend(backend).generate_sql(model_name, prompt, beam_width, max_length)
    kwargs.get("show_confidence") and "confidence_score" in result
    return result
