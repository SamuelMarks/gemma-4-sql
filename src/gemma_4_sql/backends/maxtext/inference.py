"""MaxText-specific inference logic."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_inference import (
    BeamSequenceArrayProtocol,
    BoundedLRUCache,
    compute_jax_step_probs,
    expand_jax_beams,
    normalize_logprob_confidence,
    run_array_beam_search,
    run_inference_wrapper,
)
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import (
    InferenceResultDict,
    JaxArrayProtocol,
    JSONValue,
    MaxTextCausalLMProtocol,
    extract_str_kwarg,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _JaxJitModuleProtocol(Protocol):
    """Protocol for a `jax` module exposing `.jit`."""

    def jit[ArrayT: BeamSequenceArrayProtocol](
        self,
        fun: Callable[[ArrayT, int], tuple[ArrayT, ArrayT]],
        *,
        static_argnums: tuple[int, ...],
    ) -> Callable[[ArrayT, int], tuple[ArrayT, ArrayT]]:
        """JIT-compile a step probability function."""
        ...


jax = optional_import("jax")
jnp = optional_import("jax.numpy")
_raw_gemma4 = optional_attr("maxtext.models.gemma4", "Gemma4Model")
Gemma4Model: Callable[..., MaxTextCausalLMProtocol] | None = _raw_gemma4 if callable(_raw_gemma4) else None  # ty: ignore[invalid-assignment]

MAX_MODEL_CACHE_SIZE = 4
_MODEL_CACHE: BoundedLRUCache[MaxTextCausalLMProtocol | Callable[..., JaxArrayProtocol]] = BoundedLRUCache(maxsize=MAX_MODEL_CACHE_SIZE)


def _get_or_create_maxtext_model(
    model_name: str,
) -> MaxTextCausalLMProtocol | Callable[..., JaxArrayProtocol]:
    """Retrieve a cached MaxText model instance or initialize and cache a new one.

    Args:
        model_name: Target model name.

    Returns:
        Instantiated MaxText model.

    """
    cached = _MODEL_CACHE.get_model(model_name)
    if cached is not None:
        return cached
    assert Gemma4Model is not None
    model = Gemma4Model(model_name)
    _MODEL_CACHE.put_model(model_name, model)
    return model


def _beam_search_step[ArrayT: BeamSequenceArrayProtocol](
    seq: ArrayT,
    score: float,
    model_apply_fn: Callable[[ArrayT], ArrayT],
    beam_width: int,
) -> list[tuple[ArrayT, float]]:
    """Expand a single partial token sequence into `beam_width` candidate beams using log-softmax top-k selection.

    Args:
        seq: Current token ID sequence tensor of shape `(1, current_seq_len)`.
        score: Cumulative log-probability score of `seq`.
        model_apply_fn: Callable mapping `seq` to vocabulary logits.
        beam_width: Number of top candidate token extensions to retain.

    Returns:
        List of `(expanded_sequence, updated_cumulative_score)` tuples of length `beam_width`.

    """
    logits = model_apply_fn(seq)
    if isinstance(jax, _JaxJitModuleProtocol):

        def _step_probs(l_arr: ArrayT, bw: int) -> tuple[ArrayT, ArrayT]:
            """Compute step probabilities for JIT compilation."""
            return compute_jax_step_probs(l_arr, bw, jax, jnp)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]

        step_fn = jax.jit(_step_probs, static_argnums=(1,))
        (top_indices, top_probs) = step_fn(logits, beam_width)
    else:
        (top_indices, top_probs) = compute_jax_step_probs(logits, beam_width, jax, jnp)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
    return expand_jax_beams(seq, score, top_indices, top_probs, beam_width, jnp)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]


def maxtext_beam_search[ArrayT: BeamSequenceArrayProtocol](
    model_apply_fn: Callable[[ArrayT], ArrayT],
    input_ids: ArrayT,
    beam_width: int,
    max_length: int,
    eos_token_id: int,
) -> tuple[ArrayT, float]:
    """Execute MaxText beam search decoding over token sequences.

    Args:
        model_apply_fn: Callable mapping token sequences to output logits.
        input_ids: Initial prompt token ID array.
        beam_width: Number of active beams to retain per step.
        max_length: Maximum number of new tokens to decode.
        eos_token_id: End-of-sequence token ID.

    Returns:
        Tuple of `(best_token_sequence, cumulative_log_probability)`.

    """
    return run_array_beam_search(
        input_ids,
        beam_width,
        max_length,
        eos_token_id,
        lambda seq, score, bw: _beam_search_step(seq, score, model_apply_fn, bw),
    )


def _execute_generate(
    model_name: str,
    input_tokens: list[int],
    beam_width: int,
    max_length: int,
    eos_token_id: int,
    tokenizer: SQLTokenizer,
) -> tuple[str, str, float]:
    """Execute the generation logic.

    Args:
        model_name: The name of the model.
        input_tokens: The list of input tokens.
        beam_width: The beam width for search.
        max_length: The maximum length for generation.
        eos_token_id: The end-of-sequence token ID.
        tokenizer: The tokenizer instance.

    Returns:
        A tuple of `(status, clean_sql, confidence_score)`.

    """
    assert jax is not None
    assert jnp is not None
    logger.info("Generating with MaxText: %s", model_name)
    input_ids: JaxArrayProtocol = jnp.array([input_tokens], dtype=jnp.int32)
    model = _get_or_create_maxtext_model(model_name)
    if isinstance(model, MaxTextCausalLMProtocol):
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, input_ids)

        def _apply_with_params(seq_in: JaxArrayProtocol) -> JaxArrayProtocol:
            """Apply stateless MaxText model with initialized parameters."""
            res: JaxArrayProtocol = model.apply(params, seq_in)  # pyright: ignore[reportArgumentType,reportAssignmentType]  # ty: ignore[invalid-assignment, invalid-argument-type]
            return res

        apply_fn: Callable[[JaxArrayProtocol], JaxArrayProtocol] = _apply_with_params
    else:
        apply_fn = model  # pyright: ignore[reportAssignmentType]
        with contextlib.suppress(AttributeError):
            apply_fn = model.apply  # pyright: ignore[reportAttributeAccessIssue,reportFunctionMemberAccess,reportAssignmentType]  # ty: ignore[unresolved-attribute]
    (output_ids, logprob_sum) = maxtext_beam_search(apply_fn, input_ids, beam_width, max_length, eos_token_id)
    raw_list = output_ids[0].tolist()
    tok_list = [int(t) for t in raw_list if isinstance(t, (int, float, bool))] if isinstance(raw_list, list) else [int(raw_list)]
    sql = tokenizer.decode(tok_list)
    out_len = int(output_ids.shape[1])
    confidence_score = normalize_logprob_confidence(logprob_sum, max(1, out_len - len(input_tokens)))
    return "success", sql, confidence_score


def generate_sql(
    model_name: str,
    prompt: str,
    beam_width: int = 3,
    max_length: int = 50,
    **kwargs: JSONValue,
) -> InferenceResultDict:
    """Generate a SQL query from a natural language prompt using MaxText.

    Args:
        model_name: The name of the model to use.
        prompt: The natural language prompt.
        beam_width: Number of beams for search.
        max_length: Maximum number of tokens to generate.
        **kwargs: Extra arguments.

    Returns:
        A dictionary containing the generated SQL.

    Raises:
        DependencyMissingError: If MaxText dependencies are missing.

    """
    if jax is None or jnp is None or Gemma4Model is None:
        raise DependencyMissingError("MaxText dependencies are missing.")

    tokenizer_name = extract_str_kwarg(kwargs, "tokenizer_name")
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    input_tokens = tokenizer.encode(prompt)
    eos_token_id = tokenizer.eos_token_id

    def _execute() -> tuple[str, float]:
        """Run MaxText beam search generation and return `(sql, confidence_score)`."""
        _status, sql, confidence_score = _execute_generate(
            model_name,
            input_tokens,
            beam_width,
            max_length,
            eos_token_id,
            tokenizer,
        )
        return (sql, confidence_score)

    wrapper_kwargs: dict[str, StatusValue] = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None}
    return run_inference_wrapper("maxtext", model_name, prompt, beam_width, _execute, **wrapper_kwargs)
