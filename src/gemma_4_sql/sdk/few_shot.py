"""SDK Few-Shot module for dynamic prompting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue


def build_few_shot_prompt(model_name: str, prompt: str, examples: list[dict[str, str]], backend: str = "jax", **_kwargs: JSONValue) -> JSONDict:
    """Build a dynamic few-shot prompt.

    Args:
        model_name: The name of the target model.
        prompt: The input text prompt.
        examples: A sequence of examples.
        backend: The backend framework to use.
        **_kwargs: Optional parameters for prompt formatting.

    Returns:
        A dictionary containing the results.
    """
    status = f"success_{backend}_few_shot"
    formatted_examples = "\n".join([f"Input: {ex.get('input', '')}\nOutput: {ex.get('output', '')}" for ex in examples])
    full_prompt = f"{formatted_examples}\nInput: {prompt}\nOutput: "
    return {"backend": backend, "model": model_name, "few_shot_prompt": full_prompt, "status": status}
