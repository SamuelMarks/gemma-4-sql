"""Keras-specific inference logic with BeamSampler and dynamic confidence scoring."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from gemma_4_sql._types_models import KerasSamplerProtocol, KerasScoreType
from gemma_4_sql._types_tensors import NestedScalarList, TensorProtocol
from gemma_4_sql.backends.common_inference import (
    BoundedLRUCache,
    make_model_cache_key,
    normalize_logprob_confidence,
    run_inference_wrapper,
)
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError, StatusValue
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import (
    InferenceResultDict,
    JSONValue,
    KerasCausalLMProtocol,
    KerasLayerProtocol,
    NDArrayProtocol,
    TfTensorProtocol,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _GenerateOnlyKerasModelProtocol(Protocol):
    """Protocol for minimal Keras generation models exposing `.generate(prompt, max_length=...)`."""

    def generate(
        self,
        prompt: str,
        *,
        max_length: int = 50,
    ) -> (
        TensorProtocol
        | TfTensorProtocol
        | Sequence[int]
        | Sequence[str]
        | Mapping[str, KerasScoreType]
        | tuple[str, KerasScoreType]
        | str
        | int
    ):
        """Generate SQL text or structured output.

        Args:
            prompt: Input prompt string.
            max_length: Maximum sequence length.

        Returns:
            Generated output.

        """
        ...


@runtime_checkable
class _GemmaCausalLMLoaderProtocol(Protocol):
    """Protocol for `keras_nlp.models.GemmaCausalLM` exposing `.from_preset(model_name)`."""

    def from_preset(self, model_name: str, /) -> KerasCausalLMProtocol | _GenerateOnlyKerasModelProtocol:
        """Load a `GemmaCausalLM` model from a preset identifier.

        Args:
            model_name: Preset identifier string.

        Returns:
            Loaded `KerasCausalLMProtocol` instance.

        """
        ...


@runtime_checkable
class _ToListOnlyScoreProtocol(Protocol):
    """Protocol for score arrays or tensors exposing `.tolist()`."""

    def tolist(self) -> NestedScalarList:
        """Convert score tensor into nested Python scalar list.

        Returns:
            Nested float/int list or scalar.

        """
        ...


@runtime_checkable
class _CompileSamplerModelProtocol(Protocol):
    """Protocol for KerasNLP models supporting `.compile(sampler=...)`."""

    def compile(self, *, sampler: KerasSamplerProtocol) -> None:
        """Compile model with `sampler`.

        Args:
            sampler: Configured `KerasSamplerProtocol`.

        """
        ...


@runtime_checkable
class _MutableSamplerAttrProtocol(Protocol):
    """Protocol for KerasNLP models exposing mutable `.sampler`."""

    sampler: KerasSamplerProtocol | str | Callable[..., TensorProtocol] | None


@runtime_checkable
class _LastScoresModelProtocol(Protocol):
    """Protocol for KerasNLP models exposing `.last_scores`."""

    @property
    def last_scores(self) -> KerasScoreType:
        """Provide scores recorded from the last generation call.

        Returns:
            Recorded scores or `None`.

        """
        ...


type _BeamSamplerInstanceProtocol = KerasSamplerProtocol
type KerasScoreInput = (
    TfTensorProtocol
    | NDArrayProtocol
    | _ToListOnlyScoreProtocol
    | NestedScalarList
    | float
    | int
    | str
    | Sequence[float | int | str | Sequence[float | int | str | None] | None]
    | None
)

keras = optional_import("keras")
tf = optional_import("tensorflow")
GemmaCausalLM: _GemmaCausalLMLoaderProtocol | None = import_optional_symbol(
    "keras_nlp.models",
    "GemmaCausalLM",
    _GemmaCausalLMLoaderProtocol,
)

_MODEL_CACHE: BoundedLRUCache[KerasCausalLMProtocol | _GenerateOnlyKerasModelProtocol] = BoundedLRUCache(maxsize=4)


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

    Args:
        scores: Sequence log probabilities, token probabilities, or scalar score.
        num_tokens: Number of generated tokens `L`.

    Returns:
        Confidence score between `0.0` and `1.0`.

    """
    if num_tokens <= 0:
        return 0.0
    resolved: KerasScoreInput = (
        scores.tolist() if isinstance(scores, _ToListOnlyScoreProtocol) and callable(scores.tolist) else scores
    )

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
    model: KerasCausalLMProtocol
    | _GenerateOnlyKerasModelProtocol
    | _CompileSamplerModelProtocol
    | _MutableSamplerAttrProtocol
    | KerasLayerProtocol
    | str
    | None,
    beam_width: int,
) -> _BeamSamplerInstanceProtocol | None:
    """Configure KerasNLP `BeamSampler` on a causal language model.

    Args:
        model: Target KerasNLP `CausalLM` instance.
        beam_width: Number of beams for beam search.

    Returns:
        The instantiated sampler, or `None` if unavailable.

    """

    @runtime_checkable
    class _KerasNlpSamplersNamespaceProtocol(Protocol):
        """Protocol for `keras_nlp.samplers` exposing `BeamSampler`."""

        @property
        def BeamSampler(self) -> Callable[..., _BeamSamplerInstanceProtocol | str | None] | None:
            """Provide `BeamSampler` constructor or `None`."""
            ...

    @runtime_checkable
    class _KerasNlpSamplersPkgProtocol(Protocol):
        """Protocol for `keras_nlp` exposing `.samplers`."""

        @property
        def samplers(self) -> _KerasNlpSamplersNamespaceProtocol:
            """Provide `keras_nlp.samplers` namespace."""
            ...

    sampler: _BeamSamplerInstanceProtocol | None = None
    try:
        keras_nlp_mod = optional_import("keras_nlp", _KerasNlpSamplersPkgProtocol)
        if keras_nlp_mod is not None:
            beam_sampler_cls = keras_nlp_mod.samplers.BeamSampler
            if callable(beam_sampler_cls):
                candidate = beam_sampler_cls(num_beams=beam_width)
                if isinstance(candidate, KerasSamplerProtocol):
                    sampler = candidate
    except (ImportError, AttributeError, ValueError, TypeError) as e:
        logger.warning("KerasNLP BeamSampler could not be imported: %s", e)

    if sampler is not None:
        if isinstance(model, _CompileSamplerModelProtocol) and callable(model.compile):
            try:
                model.compile(sampler=sampler)
                return sampler
            except (AttributeError, TypeError):
                pass
        if isinstance(model, _MutableSamplerAttrProtocol):
            model.sampler = sampler

    return sampler


def generate_sql(
    model_name: str,
    prompt: str,
    beam_width: int = 3,
    max_length: int = 50,
    **kwargs: JSONValue,
) -> InferenceResultDict:
    """Generate a SQL query from a natural language prompt using Keras.

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
        """Run Keras model preset loading, BeamSampler configuration, and generation.

        Returns:
            Tuple of `(generated_sql, confidence_score)`.

        Raises:
            ImportError: If `keras_nlp.models.GemmaCausalLM` is unavailable.
            InferenceError: If Keras generation yields an empty SQL sequence.

        """
        logger.info("Generating with Keras %s (beam_width=%d)", model_name, beam_width)
        gemma_causal_lm_cls = (
            GemmaCausalLM
            if GemmaCausalLM is not None
            else import_optional_symbol("keras_nlp.models", "GemmaCausalLM", _GemmaCausalLMLoaderProtocol)
        )
        if gemma_causal_lm_cls is None:
            raise ImportError("keras_nlp.models.GemmaCausalLM is not available.")
        cache_key = make_model_cache_key(
            model_name,
            backend_name="keras",
            factories=(gemma_causal_lm_cls.from_preset,),
        )
        model = _MODEL_CACHE.get_model(cache_key) or gemma_causal_lm_cls.from_preset(model_name)
        _MODEL_CACHE.put_model(cache_key, model)

        configure_beam_sampler(model, beam_width)
        output = model.generate(prompt, max_length=max_length)

        scores: KerasScoreInput = None
        if isinstance(output, dict):
            raw_text = str(output.get("text", ""))
            raw_scores = output.get("scores") or output.get("token_probabilities")
            scores = raw_scores
        elif isinstance(output, (tuple, list)) and len(output) >= 2:
            raw_text = str(output[0])
            scores = output[1]
        elif isinstance(output, str):
            raw_text = output
            if isinstance(model, _LastScoresModelProtocol):
                scores = model.last_scores
        else:
            raw_text = str(output)

        sql = (
            raw_text[len(prompt) :].strip()
            if prompt and raw_text.startswith(prompt)
            else raw_text.replace(prompt, "", 1).strip()
        )
        if not sql:
            raise InferenceError("Keras generation yielded an empty SQL sequence.")

        tokenizer = SQLTokenizer(model_name=model_name)
        num_tokens = max(1, len(tokenizer.encode(sql)))
        confidence_score = compute_keras_confidence(scores, num_tokens)
        return (sql, confidence_score)

    wrapper_kwargs: dict[str, StatusValue] = {
        k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None
    }
    return run_inference_wrapper("keras", model_name, prompt, beam_width, _execute, **wrapper_kwargs)
