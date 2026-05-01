"""
SDK module for Dynamic Few-Shot Prompting.
"""

from __future__ import annotations

from typing import Any


def build_few_shot_prompt(
    model_name: str,
    prompt: str,
    examples: list[dict[str, str]],
    backend: str = "jax",
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Builds a dynamic few-shot prompt.

    Args:
        model_name: The name of the model to use.
        prompt: The new user prompt.
        examples: A list of dictionaries representing few-shot examples.
        backend: The backend framework ('jax', 'keras', 'maxtext', 'pytorch').
        **kwargs: Additional parameters.

    Returns:
        A dictionary containing the generated few-shot prompt.
    """
    from gemma_4_sql.sdk.registry import get_backend

    return get_backend(backend).build_few_shot_prompt(
        model_name, prompt, examples, **kwargs
    )
