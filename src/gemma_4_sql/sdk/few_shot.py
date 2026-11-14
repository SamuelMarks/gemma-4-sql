"""SDK Few-Shot module for dynamic prompting and multimodal few-shot SQL generation."""

from __future__ import annotations

import logging

from gemma_4_sql.constants import DEFAULT_BACKEND
from gemma_4_sql.sdk.registry import get_backend, validate_backend_name
from gemma_4_sql.sdk.sql_lexer import extract_word_tokens
from gemma_4_sql.type_hints import (
    FewShotPromptResultDict,
    FewShotResultDict,
    JSONValue,
    extract_float_kwarg,
    extract_int_kwarg,
)

logger = logging.getLogger(__name__)
_NL = """
"""


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
        example_pool: Candidate pool of demonstration examples with `'input'` and `'output'` keys.
        top_k: Maximum number of examples to select.
        max_tokens: Approximate maximum token budget (1 token ~= 4 chars).

    Returns:
        List of prioritized, budgeted demonstration examples.

    """
    if not example_pool:
        return []

    prompt_words = set(extract_word_tokens(prompt))

    def _score(ex: dict[str, str]) -> float:
        """Compute keyword overlap fraction between example input and prompt.

        Args:
            ex: Demonstration example dictionary.

        Returns:
            Normalized overlap score in `[0.0, 1.0]`.

        """
        ex_words = set(extract_word_tokens(ex.get("input", "")))
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
    backend: str = DEFAULT_BACKEND,
    *,
    top_k: int | None = None,
    max_tokens: int = 2048,
    image_path: str | None = None,
    audio_path: str | None = None,
    modality: str = "text",
    **_kwargs: JSONValue,
) -> FewShotPromptResultDict:
    """Build a dynamic few-shot prompt with lightweight backend validation and multimodal placeholders.

    Args:
        model_name: The name of the target model.
        prompt: The input text prompt.
        examples: A sequence of demonstration examples.
        backend: The backend framework identifier.
        top_k: Optional maximum number of examples to select.
        max_tokens: Approximate token budget for examples.
        image_path: Optional path to a schema diagram or ERD image.
        audio_path: Optional path to a recorded query audio file.
        modality: Explicit modality selector (`'text'`, `'vision'`, `'audio'`, `'multimodal'`).
        **_kwargs: Optional parameters reserved for forward compatibility.

    Returns:
        A structured `FewShotPromptResultDict` containing the formatted prompt.

    Raises:
        ConfigurationError: If `backend` is not a registered backend identifier.

    """
    validate_backend_name(backend)
    active_examples: list[dict[str, str]]
    if top_k is not None and top_k > 0:
        active_examples = select_relevant_examples(prompt, examples, top_k=top_k, max_tokens=max_tokens)
    elif examples:
        char_budget = max_tokens * 4
        active_examples = []
        current_chars = 0
        for ex in examples:
            ex_chars = len(ex.get("input", "")) + len(ex.get("output", "")) + 20
            if current_chars + ex_chars <= char_budget:
                active_examples.append(ex)
                current_chars += ex_chars
    else:
        active_examples = []

    status = f"success_{backend}_few_shot"
    formatted_examples = _NL.join([
        f"Input: {ex.get('input', '')}{_NL}Output: {ex.get('output', '')}" for ex in active_examples
    ])
    if formatted_examples:
        full_prompt = f"{formatted_examples}{_NL}Input: {prompt}{_NL}Output: "
    else:
        full_prompt = f"Input: {prompt}{_NL}Output: "

    effective_img = None if modality == "audio" else image_path
    effective_aud = None if modality == "vision" else audio_path
    if effective_img is not None or effective_aud is not None:
        from gemma_4_sql.backends.common_multimodal import format_multimodal_prompt

        mm = format_multimodal_prompt(
            full_prompt,
            has_image=effective_img is not None,
            has_audio=effective_aud is not None,
        )
        full_prompt = str(mm.get("prompt", full_prompt))

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
    backend: str = DEFAULT_BACKEND,
    top_k: int = 3,
    max_tokens: int = 2048,
    **kwargs: JSONValue,
) -> FewShotResultDict:
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
    call_kwargs = dict(kwargs)
    raw_img = call_kwargs.get("image_path")
    raw_aud = call_kwargs.get("audio_path")
    raw_mod = call_kwargs.get("modality")
    img_path = str(raw_img) if isinstance(raw_img, str) else None
    aud_path = str(raw_aud) if isinstance(raw_aud, str) else None
    mod_str = str(raw_mod) if isinstance(raw_mod, str) else "text"

    prompt_res = build_few_shot_prompt(
        model_name=model_name,
        prompt=prompt,
        examples=examples,
        backend=backend,
        top_k=top_k,
        max_tokens=max_tokens,
        image_path=img_path,
        audio_path=aud_path,
        modality=mod_str,
    )
    few_shot_prompt = str(prompt_res["few_shot_prompt"])

    backend_impl = get_backend(backend)
    beam_width = extract_int_kwarg(call_kwargs, "beam_width", 3)
    max_length = extract_int_kwarg(call_kwargs, "max_length", 50)
    call_kwargs.pop("beam_width", None)
    call_kwargs.pop("max_length", None)
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
        "sql": str(gen_res.get("sql", "")),
        "confidence_score": extract_float_kwarg(gen_res, "confidence_score", 0.0),
        "status": str(gen_res.get("status", "success")),
        "num_examples": extract_int_kwarg(prompt_res, "num_examples", 0),
    }


__all__ = ["build_few_shot_prompt", "generate_few_shot_sql", "select_relevant_examples"]
