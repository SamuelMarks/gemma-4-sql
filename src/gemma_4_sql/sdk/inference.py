"""SDK Inference module with text and multimodal generation support."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from gemma_4_sql.type_hints import JSONDict


def generate(
    model_name: str,
    prompt: str,
    backend: str = "jax",
    beam_width: int = 3,
    max_length: int = 50,
    image_path: str | Path | None = None,
    audio_path: str | Path | None = None,
    modality: str = "text",
    **kwargs: object,
) -> JSONDict:
    """Generate a SQL query from a natural language prompt using Beam Search.

    Args:
        model_name: The name of the target model.
        prompt: The input text prompt.
        backend: The backend framework to use.
        beam_width: The number of beams for beam search.
        max_length: The maximum length of the sequence.
        image_path: Optional path to schema diagram or ERD image.
        audio_path: Optional path to recorded natural language query audio.
        modality: Explicit modality selector ('text', 'vision', 'audio', 'multimodal').
        **kwargs: Advanced generation parameters (e.g., temperature, top_p, show_confidence).

    Returns:
        A dictionary containing the generated SQL and generation metadata.
    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    backend_impl = get_backend(backend)
    result = backend_impl.generate_sql(
        model_name,
        prompt,
        beam_width=beam_width,
        max_length=max_length,
        image_path=image_path,
        audio_path=audio_path,
        modality=modality,
        **kwargs,
    )
    return result


# Alias for generate adhering to backend protocol naming
generate_sql = generate

__all__ = ["generate", "generate_sql"]
