"""SDK Few-Shot module for dynamic prompting."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue

logger = logging.getLogger(__name__)


def select_relevant_examples(
    prompt: str,
    example_pool: list[dict[str, str]],
    top_k: int = 3,
    max_tokens: int = 2048,
) -> list[dict[str, str]]:
    """Select top-k most relevant demonstration examples with token budgeting.

    Uses keyword set overlap scoring to prioritize examples most similar to the target prompt,
    while pruning candidates to stay within context length budgets.

    Args:
        prompt: Target natural language question.
        example_pool: Candidate pool of demonstration examples with 'input' and 'output' keys.
        top_k: Maximum number of examples to select.
        max_tokens: Approximate maximum token budget (1 token ~= 4 chars).

    Returns:
        List of prioritized, budgeted demonstration examples.
    """
    if not example_pool:
        return []

    prompt_words = set(prompt.lower().split())

    def _score(ex: dict[str, str]) -> float:
        """Compute keyword overlap fraction between example input and prompt."""
        ex_words = set(ex.get("input", "").lower().split())
        overlap = len(prompt_words.intersection(ex_words))
        return float(overlap) / max(1.0, float(len(ex_words)))

    scored_pool = sorted(example_pool, key=_score, reverse=True)

    selected: list[dict[str, str]] = []
    current_chars = 0
    char_budget = max_tokens * 4

    for ex in scored_pool[:top_k]:
        ex_chars = len(ex.get("input", "")) + len(ex.get("output", "")) + 20
        if current_chars + ex_chars <= char_budget:
            selected.append(ex)
            current_chars += ex_chars

    return selected


def build_few_shot_prompt(
    model_name: str,
    prompt: str,
    examples: list[dict[str, str]],
    backend: str = "jax",
    top_k: int | None = None,
    max_tokens: int = 2048,
    **_kwargs: JSONValue,
) -> JSONDict:
    """Build a dynamic few-shot prompt with optional example selection.

    Args:
        model_name: The name of the target model.
        prompt: The input text prompt.
        examples: A sequence of examples.
        backend: The backend framework to use.
        top_k: Optional maximum number of examples to select.
        max_tokens: Approximate token budget for examples.
        **_kwargs: Optional parameters for prompt formatting.

    Returns:
        A dictionary containing the results.
    """
    if top_k is not None and top_k > 0:
        active_examples = select_relevant_examples(prompt, examples, top_k=top_k, max_tokens=max_tokens)
    else:
        active_examples = examples

    status = f"success_{backend}_few_shot"
    formatted_examples = "\n".join([f"Input: {ex.get('input', '')}\nOutput: {ex.get('output', '')}" for ex in active_examples])
    if formatted_examples:
        full_prompt = f"{formatted_examples}\nInput: {prompt}\nOutput: "
    else:
        full_prompt = f"Input: {prompt}\nOutput: "

    return {
        "backend": backend,
        "model": model_name,
        "few_shot_prompt": full_prompt,
        "status": status,
        "num_examples": len(active_examples),
    }


def generate_few_shot_sql(
    model_name: str,
    prompt: str,
    examples: list[dict[str, str]],
    backend: str = "jax",
    top_k: int = 3,
    max_tokens: int = 2048,
    **kwargs: JSONValue,
) -> JSONDict:
    """Generate SQL query using dynamic few-shot prompting and backend inference.

    Args:
        model_name: Target model identifier.
        prompt: Input natural language prompt.
        examples: Candidate demonstration examples.
        backend: Target inference backend.
        top_k: Maximum number of examples to include.
        max_tokens: Approximate token budget for examples.
        **kwargs: Additional parameters passed to backend inference.

    Returns:
        Dictionary containing generation output, confidence score, and used prompt.
    """
    prompt_res = build_few_shot_prompt(
        model_name=model_name,
        prompt=prompt,
        examples=examples,
        backend=backend,
        top_k=top_k,
        max_tokens=max_tokens,
    )
    few_shot_prompt = str(prompt_res["few_shot_prompt"])

    from gemma_4_sql.sdk.registry import get_backend

    backend_impl = get_backend(backend)
    call_kwargs = dict(kwargs)
    beam_width = int(str(call_kwargs.pop("beam_width", 3))) if "beam_width" in call_kwargs else 3
    max_length = int(str(call_kwargs.pop("max_length", 50))) if "max_length" in call_kwargs else 50
    gen_res = backend_impl.generate_sql(
        model_name,
        few_shot_prompt,
        beam_width=beam_width,
        max_length=max_length,
        **call_kwargs,
    )

    return {
        "backend": backend,
        "model": model_name,
        "prompt": prompt,
        "few_shot_prompt": few_shot_prompt,
        "sql": gen_res.get("sql", ""),
        "confidence_score": gen_res.get("confidence_score", 0.0),
        "status": gen_res.get("status", "success"),
        "num_examples": prompt_res.get("num_examples", 0),
    }
