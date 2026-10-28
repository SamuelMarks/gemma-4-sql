"""Keras-specific inference logic with BeamSampler and dynamic confidence scoring."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_inference import (
    BoundedLRUCache,
    make_model_cache_key,
    normalize_logprob_confidence,
    run_inference_wrapper,
)
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError, StatusValue
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import (
    InferenceResultDict,
    JSONValue,
    KerasCausalLMProtocol,
    NDArrayProtocol,
    TfTensorProtocol,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _GemmaCausalLMLoaderProtocol(Protocol):
    """Protocol for `keras_nlp.models.GemmaCausalLM` exposing `.from_preset(model_name)`."""

    def from_preset(self, model_name: str, /) -> KerasCausalLMProtocol:
        """Load a `GemmaCausalLM` model from a preset identifier."""
        ...


@runtime_checkable
class _BeamSamplerInstanceProtocol(Protocol):
    """Protocol for an instantiated KerasNLP `BeamSampler`."""


type KerasScoreInput = TfTensorProtocol | NDArrayProtocol | float | int | str | Sequence[float | int | str | Sequence[float | int | str | None] | None] | None


keras = optional_import("keras")
tf = optional_import("tensorflow")
GemmaCausalLM: _GemmaCausalLMLoaderProtocol | None = optional_attr(
    "keras_nlp.models",
    "GemmaCausalLM",
    _GemmaCausalLMLoaderProtocol,
)

_MODEL_CACHE: BoundedLRUCache[KerasCausalLMProtocol] = BoundedLRUCache(maxsize=4)


def _extract_flat_scores(scores: KerasScoreInput) -> list[float]:
    """Extract a flattened list of float scores from nested containers.

    Args:
        scores: Input scores structure (list, tuple, or scalar).

    Returns:
        Flattened list of floating-point values.

    """
    flat: list[float] = []
    stack: list[KerasScoreInput] = [scores]
    while stack:
        curr = stack.pop(0)
        if isinstance(curr, (list, tuple)):
            stack.extend(curr)
        elif isinstance(curr, (int, float)):
            flat.append(float(curr))
    return flat


def compute_keras_confidence(scores: KerasScoreInput, num_tokens: int) -> float:
    """Compute length-normalized sequence confidence score from generation scores or probabilities.

    Mathematical formulation:
        For log-probabilities: `confidence = exp((1 / max(1, L)) * sum(log_probs))`
        For linear probabilities: `confidence = (1 / max(1, L)) * sum(probs)`
    Bounded in `[0.0, 1.0]`.

    Args:
        scores: Sequence log probabilities, token probabilities, or scalar score.
        num_tokens: Number of generated tokens `L`.

    Returns:
        Confidence score between `0.0` and `1.0`.

    """
    if num_tokens <= 0:
        return 0.0
    resolved: KerasScoreInput = scores
    with contextlib.suppress(AttributeError, TypeError):
        resolved = resolved.tolist()  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue,reportAssignmentType]  # ty: ignore[invalid-assignment, unresolved-attribute]

    if isinstance(resolved, (list, tuple)):
        flat_scores = _extract_flat_scores(resolved)
        if not flat_scores:
            return normalize_logprob_confidence(0.0, num_tokens)
        avg = sum(flat_scores) / max(1, len(flat_scores))
        if avg <= 0.0:
            return normalize_logprob_confidence(flat_scores, len(flat_scores))
        return max(0.0, min(1.0, avg))

    if isinstance(resolved, (int, float)):
        val = float(resolved)
        if val <= 0.0:
            return normalize_logprob_confidence(val, num_tokens)
        return max(0.0, min(1.0, val))

    return normalize_logprob_confidence(0.0, num_tokens)


def configure_beam_sampler(
    model: KerasCausalLMProtocol,
    beam_width: int,
) -> _BeamSamplerInstanceProtocol | str | None:
    """Configure KerasNLP `BeamSampler` on a causal language model.

    Args:
        model: Target KerasNLP `CausalLM` instance.
        beam_width: Number of beams for beam search.

    Returns:
        The instantiated sampler, or `None` if unavailable.

    """
    sampler: _BeamSamplerInstanceProtocol | str | None = None
    try:
        import keras_nlp

        beam_sampler_cls = keras_nlp.samplers.BeamSampler
        if callable(beam_sampler_cls):
            sampler = beam_sampler_cls(num_beams=beam_width)
    except (ImportError, AttributeError, ValueError, TypeError) as e:
        logger.warning("KerasNLP BeamSampler could not be imported: %s", e)

    if sampler is not None:
        try:
            model.compile(sampler=sampler)  # pyright: ignore[reportArgumentType]
        except (AttributeError, TypeError):
            with contextlib.suppress(AttributeError, TypeError):
                model.sampler = sampler  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[invalid-assignment]

    return sampler


def generate_sql(
    model_name: str,
    prompt: str,
    beam_width: int = 3,
    max_length: int = 50,
    **kwargs: JSONValue,
) -> InferenceResultDict:
    """Generate a SQL query from a natural language prompt using Keras.

    Configures `BeamSampler` with the requested `beam_width`, executes generation,
    extracts token scores, and dynamically computes sequence confidence.

    Args:
        model_name: The name or preset of the target model.
        prompt: The input natural language text prompt.
        beam_width: The number of beams for beam search.
        max_length: The maximum length of the sequence.
        **kwargs: Advanced generation parameters.

    Returns:
        A dictionary containing the generated SQL query and generation metadata.

    Raises:
        DependencyMissingError: If Keras or TensorFlow dependencies are missing.

    """
    if keras is None or tf is None:
        raise DependencyMissingError("Keras dependencies are missing.")

    def _execute() -> tuple[str, float]:
        """Run Keras model preset loading, BeamSampler configuration, and generation."""
        logger.info("Generating with Keras %s (beam_width=%d)", model_name, beam_width)
        gemma_causal_lm_cls = GemmaCausalLM if GemmaCausalLM is not None else optional_attr("keras_nlp.models", "GemmaCausalLM", _GemmaCausalLMLoaderProtocol)
        if gemma_causal_lm_cls is None:
            raise ImportError("keras_nlp.models.GemmaCausalLM is not available.")
        cache_key = make_model_cache_key(
            model_name,
            backend_name="keras",
            factories=(gemma_causal_lm_cls, gemma_causal_lm_cls.from_preset),
        )
        model = _MODEL_CACHE.get_model(cache_key) or gemma_causal_lm_cls.from_preset(model_name)
        _MODEL_CACHE.put_model(cache_key, model)

        configure_beam_sampler(model, beam_width)
        output = model.generate(prompt, max_length=max_length)

        scores: KerasScoreInput = None
        if isinstance(output, dict):
            raw_text = str(output.get("text", ""))
            raw_scores = output.get("scores") or output.get("token_probabilities")
            scores = raw_scores  # pyright: ignore[reportAssignmentType]
        elif isinstance(output, (tuple, list)) and len(output) >= 2:
            raw_text = str(output[0])
            scores = output[1]  # pyright: ignore[reportAssignmentType]
        elif isinstance(output, str):
            raw_text = output
            with contextlib.suppress(AttributeError):
                scores = model.last_scores  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        else:
            raw_text = str(output)

        sql = raw_text[len(prompt) :].strip() if prompt and raw_text.startswith(prompt) else raw_text.replace(prompt, "", 1).strip()
        if not sql:
            raise InferenceError("Keras generation yielded an empty SQL sequence.")

        tokenizer = SQLTokenizer(model_name=model_name)
        num_tokens = max(1, len(tokenizer.encode(sql)))
        confidence_score = compute_keras_confidence(scores, num_tokens)
        return (sql, confidence_score)

    wrapper_kwargs: dict[str, StatusValue] = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None}
    return run_inference_wrapper("keras", model_name, prompt, beam_width, _execute, **wrapper_kwargs)
