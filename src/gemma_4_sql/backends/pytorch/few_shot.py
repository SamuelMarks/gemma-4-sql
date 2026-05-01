"""
PyTorch-specific dynamic few-shot prompting logic.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:
    torch = None


def build_few_shot_prompt(
    model_name: str, prompt: str, examples: list[dict[str, str]], **kwargs: Any
) -> dict[str, Any]:
    """
    Builds a dynamic few-shot prompt using PyTorch backend.

    Args:
        model_name: The name of the model.
        prompt: The user prompt.
        examples: List of example dictionaries.
        **kwargs: Additional parameters.

    Returns:
        A dictionary containing the generated few-shot prompt and status.
    """
    if torch is not None:
        status = "success_pytorch_few_shot"
        formatted_examples = "\n".join(
            [
                f"Input: {ex.get('input', '')}\nOutput: {ex.get('output', '')}"
                for ex in examples
            ]
        )
        full_prompt = f"{formatted_examples}\nInput: {prompt}\nOutput: "
    else:
        status = "mocked_missing_pytorch"  # pragma: no cover
        full_prompt = "Fallback few-shot prompt"  # pragma: no cover

    return {
        "backend": "pytorch",
        "model": model_name,
        "few_shot_prompt": full_prompt,
        "status": status,
    }
