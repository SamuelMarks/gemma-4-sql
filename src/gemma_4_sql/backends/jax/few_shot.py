"""JAX-specific dynamic few-shot prompting logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue

try:
    import jax
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    jax = None


def build_few_shot_prompt(model_name: str, prompt: str, examples: list[dict[str, str]], **_kwargs: JSONValue) -> JSONDict:
    """Build a dynamic few-shot prompt using JAX backend.

    Args:
    ----
        model_name: The name of the model.
        prompt: The user prompt.
        examples: List of example dictionaries (e.g., {"input": "...", "output": "..."}).
        **kwargs: Additional parameters.

    Returns:
    -------
        A dictionary containing the generated few-shot prompt and status.

    """
    if jax is not None:
        status = "success_jax_few_shot"
        formatted_examples = "\n".join([f"Input: {ex.get('input', '')}\nOutput: {ex.get('output', '')}" for ex in examples])
        full_prompt = f"{formatted_examples}\nInput: {prompt}\nOutput: "
    else:
        status = "mocked_missing_jax"
        full_prompt = "Fallback few-shot prompt"
    return {"backend": "jax", "model": model_name, "few_shot_prompt": full_prompt, "status": status}
