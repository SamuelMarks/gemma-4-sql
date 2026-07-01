"""SDK module for Dynamic Few-Shot Prompting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue


def build_few_shot_prompt(model_name: str, prompt: str, examples: list[dict[str, str]], backend: str = "jax", **kwargs: JSONValue) -> JSONDict:
    """Build a dynamic few-shot prompt.

    Args:
    ----
        model_name: The name of the model to use.
        prompt: The new user prompt.
        examples: A list of dictionaries representing few-shot examples.
        backend: The backend framework ('jax', 'keras', 'maxtext', 'pytorch').
        **kwargs: Additional parameters.

    Returns:
    -------
        A dictionary containing the generated few-shot prompt.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).build_few_shot_prompt(model_name, prompt, examples, **kwargs)
