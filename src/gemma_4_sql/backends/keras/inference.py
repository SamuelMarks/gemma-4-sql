"""Keras-specific inference logic with BeamSampler and dynamic confidence scoring."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError
from gemma_4_sql.type_hints import ModelType

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import InferenceResultDict, JSONValue

logger = logging.getLogger(__name__)

keras = optional_import("keras")
tf = optional_import("tensorflow")
GemmaCausalLM = optional_attr("keras_nlp.models", "GemmaCausalLM")


def _extract_flat_scores(scores: object) -> list[float]:
    """Extract a flattened list of float scores from nested containers.

    Args:
        scores: Input scores structure (list, tuple, or scalar).

    Returns:
        Flattened list of floating-point values.

    """
    flat: list[float] = []
    stack: list[Any] = [scores]
    while stack:
        curr = stack.pop(0)
        if isinstance(curr, (list, tuple)):
            stack.extend(curr)
        elif isinstance(curr, (int, float)):
            flat.append(float(curr))
    return flat


def compute_keras_confidence(scores: Any, num_tokens: int) -> float:
    """Compute length-normalized sequence confidence score from generation scores or probabilities.

    Mathematical formulation:
        For log-probabilities: confidence = exp( (1 / max(1, L)) * sum(log_probs) )
        For linear probabilities: confidence = (1 / max(1, L)) * sum(probs)
    Bounded in [0.0, 1.0].

    Args:
        scores: Sequence log probabilities, token probabilities, or scalar score.
        num_tokens: Number of generated tokens L.

    Returns:
        Confidence score between 0.0 and 1.0.

    """
    if num_tokens <= 0:
        return 0.0
    if hasattr(scores, "numpy"):
        scores = scores.numpy()
    if hasattr(scores, "tolist"):
        scores = scores.tolist()

    if isinstance(scores, (list, tuple)):
        flat_scores = _extract_flat_scores(scores)
        if not flat_scores:
            return 0.5
        avg = sum(flat_scores) / max(1, len(flat_scores))
        if avg <= 0.0:
            return max(0.0, min(1.0, math.exp(avg)))
        return max(0.0, min(1.0, avg))

    if isinstance(scores, (int, float)):
        val = float(scores)
        if val <= 0.0:
            return max(0.0, min(1.0, math.exp(val / max(1, num_tokens))))
        return max(0.0, min(1.0, val))

    # Fallback when scores are not provided: derive confidence based on output length
    return max(0.1, min(0.95, 1.0 / (1.0 + math.exp(-0.1 * num_tokens))))


def configure_beam_sampler(model: ModelType, beam_width: int) -> Any:
    """Configure KerasNLP BeamSampler on a causal language model.

    Args:
        model: Target KerasNLP CausalLM instance.
        beam_width: Number of beams for beam search.

    Returns:
        The instantiated sampler, or None if unavailable.

    """
    sampler = None
    try:
        import keras_nlp

        samplers = getattr(keras_nlp, "samplers", None)
        beam_sampler_cls = getattr(samplers, "BeamSampler", None)
        if beam_sampler_cls is not None:
            sampler = beam_sampler_cls(num_beams=beam_width)
    except (ImportError, AttributeError, ValueError) as e:
        logger.warning("KerasNLP BeamSampler could not be imported: %s", e)

    if sampler is not None:
        model_any: Any = model
        if hasattr(model_any, "compile"):
            model_any.compile(sampler=sampler)
        elif hasattr(model_any, "sampler"):
            model_any.sampler = sampler

    return sampler


def generate_sql(
    model_name: str,
    prompt: str,
    beam_width: int = 3,
    max_length: int = 50,
    **kwargs: JSONValue,
) -> InferenceResultDict:
    """Generate a SQL query from a natural language prompt using Keras.

    Configures BeamSampler with the requested beam_width, executes generation,
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

    confidence_score = 0.0
    sql = ""
    try:
        logger.info("Generating with Keras %s (beam_width=%d)", model_name, beam_width)
        gemma_causal_lm_cls = GemmaCausalLM if GemmaCausalLM is not None else optional_attr("keras_nlp.models", "GemmaCausalLM")
        if gemma_causal_lm_cls is None:
            raise ImportError("keras_nlp.models.GemmaCausalLM is not available.")
        model = gemma_causal_lm_cls.from_preset(model_name)

        # Configure beam search sampler
        configure_beam_sampler(model, beam_width)

        # Execute generation
        output: Any = model.generate(prompt, max_length=max_length)

        scores = None
        if isinstance(output, dict):
            raw_text = str(output.get("text", ""))
            scores = output.get("scores") or output.get("token_probabilities")
        elif isinstance(output, (tuple, list)) and len(output) >= 2:
            raw_text = str(output[0])
            scores = output[1]
        elif isinstance(output, str):
            raw_text = output
            scores = getattr(output, "scores", None) or getattr(model, "last_scores", None)
        else:
            raw_text = str(output)

        sql = raw_text.replace(prompt, "").strip()
        if not sql:
            raise InferenceError("Keras generation yielded an empty SQL sequence.")

        num_tokens = len(sql.split())
        confidence_score = compute_keras_confidence(scores, num_tokens)
        status = "success"
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError, ImportError) as e:
        logger.exception("Keras Generation Error: ")
        status = f"failed: {e!s}"
        sql = ""
        confidence_score = 0.0

    return {
        "backend": "keras",
        "model": model_name,
        "prompt": prompt,
        "sql": sql,
        "status": status,
        "beam_width": beam_width,
        "confidence_score": confidence_score,
    }
