"""SDK Few-Shot module for dynamic prompting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue


def build_few_shot_prompt(model_name: str, prompt: str, examples: list[dict[str, str]], backend: str = "jax", **_kwargs: JSONValue) -> JSONDict:
    """Build a dynamic few-shot prompt.

    Args:
    ----
        model_name: The name or path of the model.
        prompt: The natural language prompt.
        examples: List of example dictionaries (e.g., {"input": "...", "output": "..."}).
        backend: The backend framework ('jax', 'keras', 'maxtext', 'pytorch').
        **_kwargs: Additional keyword arguments.

    Returns:
    -------
        A dictionary containing the generated few-shot prompt and status.

    """
    status = f"success_{backend}_few_shot"
    formatted_examples = "\n".join([f"Input: {ex.get('input', '')}\nOutput: {ex.get('output', '')}" for ex in examples])
    full_prompt = f"{formatted_examples}\nInput: {prompt}\nOutput: "
    return {"backend": backend, "model": model_name, "few_shot_prompt": full_prompt, "status": status}
