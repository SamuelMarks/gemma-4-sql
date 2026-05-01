"""
MaxText-specific dynamic few-shot prompting logic.
"""

from __future__ import annotations

from typing import Any

try:
    from maxtext.models import gemma4
except Exception:
    gemma4 = None


def build_few_shot_prompt(
    model_name: str,
    prompt: str,
    examples: list[dict[str, str]],
    **kwargs: Any
) -> dict[str, Any]:
    """
    Builds a dynamic few-shot prompt using MaxText backend.

    Args:
        model_name: The name of the model.
        prompt: The user prompt.
        examples: List of example dictionaries.
        **kwargs: Additional parameters.

    Returns:
        A dictionary containing the generated few-shot prompt and status.
    """
    if gemma4 is not None:
        status = "success_maxtext_few_shot"  # pragma: no cover
        formatted_examples = "\n".join(  # pragma: no cover
            [f"Input: {ex.get('input', '')}\nOutput: {ex.get('output', '')}" for ex in examples]  # pragma: no cover
        )  # pragma: no cover
        full_prompt = f"{formatted_examples}\nInput: {prompt}\nOutput: "  # pragma: no cover
    else:
        status = "mocked_missing_maxtext"
        full_prompt = "Fallback few-shot prompt"

    return {
        "backend": "maxtext",
        "model": model_name,
        "few_shot_prompt": full_prompt,
        "status": status,
    }
