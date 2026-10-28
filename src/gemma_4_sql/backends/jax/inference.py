"""JAX-specific inference logic."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from gemma_4_sql.backends.common_inference import (
    BoundedLRUCache,
    call_jax_causal_lm,
    compute_jax_step_probs,
    expand_jax_beams,
    normalize_logprob_confidence,
    run_array_beam_search,
    run_inference_wrapper,
)
from gemma_4_sql.backends.common_multimodal import prepare_multimodal_inputs
from gemma_4_sql.backends.jax.gemma4.config import ModelConfig, resolve_gemma4_config
from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError, StatusValue
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    ForwardArg,
    InferenceResultDict,
    JaxArrayProtocol,
    JaxCausalLMProtocol,
    NDArrayProtocol,
    OrbaxCheckpointerProtocol,
    extract_str_kwarg,
)

logger = logging.getLogger(__name__)

Gemma4Config = ModelConfig

MAX_MODEL_CACHE_SIZE = 4
_MODEL_CACHE: BoundedLRUCache[JaxCausalLMProtocol | Callable[..., Array]] = BoundedLRUCache(maxsize=MAX_MODEL_CACHE_SIZE)
MODEL_CACHE = _MODEL_CACHE


@runtime_checkable
class _JaxJitModuleProtocol(Protocol):
    """Protocol for a `jax` module exposing `.jit`."""

    def jit(
        self,
        fun: Callable[[Array, int], tuple[Array, Array]],
        *,
        static_argnums: tuple[int, ...],
    ) -> Callable[[Array, int], tuple[Array, Array]]:
        """JIT-compile a step probability function."""
        ...


def _compute_step_probs(logits: Array, beam_width: int) -> tuple[Array, Array]:
    """Compute top-k token indices and log probabilities.

    Args:
        logits: Logits tensor from model evaluation.
        beam_width: Number of top tokens to select.

    Returns:
        Tuple of `(top_indices, top_probs)` arrays.

    """
    return compute_jax_step_probs(logits, beam_width, jax, jnp)


def _beam_search_step(
    seq: Array,
    score: float,
    model_apply_fn: Callable[[Array, Array], Array],
    beam_width: int,
) -> list[tuple[Array, float]]:
    """Process a single sequence and expand it into multiple beams.

    Args:
        seq: The sequence of token IDs so far.
        score: The cumulative log probability score.
        model_apply_fn: The model forward pass callable.
        beam_width: The number of beams for beam search.

    Returns:
        A list of tuples containing expanded sequences and their updated scores.

    """
    positions = jnp.arange(seq.shape[1])[None, :]
    logits = model_apply_fn(seq, positions)
    if isinstance(jax, _JaxJitModuleProtocol):
        step_fn = jax.jit(_compute_step_probs, static_argnums=(1,))
        (top_indices, top_probs) = step_fn(logits, beam_width)
    else:
        (top_indices, top_probs) = _compute_step_probs(logits, beam_width)
    return expand_jax_beams(seq, score, top_indices, top_probs, beam_width, jnp)


def jax_beam_search(
    model_apply_fn: Callable[[Array, Array], Array],
    input_ids: Array,
    beam_width: int,
    max_length: int,
    eos_token_id: int,
) -> tuple[Array, float]:
    """Run JAX native beam search decoding.

    Args:
        model_apply_fn: The model's forward pass function.
        input_ids: The initial input token IDs.
        beam_width: The number of beams to maintain.
        max_length: The maximum generation length.
        eos_token_id: The end-of-sequence token ID.

    Returns:
        The sequence of token IDs representing the best beam and its score.

    """
    return run_array_beam_search(
        input_ids,
        beam_width,
        max_length,
        eos_token_id,
        lambda seq, score, bw: _beam_search_step(seq, score, model_apply_fn, bw),
    )


def generate_sql(
    model_name: str,
    prompt: str,
    beam_width: int = 3,
    max_length: int = 50,
    **kwargs: BackendKwargValue | Array,
) -> InferenceResultDict:
    """Generate a SQL query from a natural language prompt using JAX.

    Args:
        model_name: The name of the model to use.
        prompt: The natural language prompt.
        beam_width: Number of beams for search.
        max_length: Maximum number of tokens to generate.
        **kwargs: Optional generation arguments.

    Returns:
        A dictionary containing the generated SQL and generation metadata.

    Raises:
        DependencyMissingError: If JAX inference dependencies are missing.

    """
    if jax is None or jnp is None or Gemma4ForCausalLM is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise DependencyMissingError("JAX inference dependencies are missing.")

    def _to_jax_tensor(
        arr: NDArrayProtocol | JaxArrayProtocol | Sequence[float] | Sequence[Sequence[float]],
    ) -> Array:
        """Convert feature array into a batched float32 JAX array."""
        return jnp.array([arr], dtype=jnp.float32)

    (prompt, pixel_values, audio_values) = prepare_multimodal_inputs(
        prompt,
        dict(kwargs),
        _to_jax_tensor,
    )

    json_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None}
    tokenizer_name = extract_str_kwarg(json_kwargs, "tokenizer_name")
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    input_tokens = tokenizer.encode(prompt)
    eos_token_id = tokenizer.eos_token_id
    strict_checkpoint = bool(kwargs.get("strict_checkpoint"))
    raise_on_error = bool(kwargs.get("raise_on_error"))

    def _execute() -> tuple[str, float]:
        """Run JAX model loading, optional Orbax restore, and beam search decoding."""
        input_ids = jnp.array([input_tokens], dtype=jnp.int32)
        cached_model = _MODEL_CACHE.get_model(model_name)
        if cached_model is not None:
            model = cached_model
        else:
            rngs = nnx.Rngs(0) if nnx is not None else None
            cfg = resolve_gemma4_config(Gemma4Config, model_name)
            model = Gemma4ForCausalLM(cfg, rngs=rngs)  # pyright: ignore[reportArgumentType]
            model_path = Path(model_name)
            if model_path.exists():
                try:
                    import orbax.checkpoint as ocp

                    checkpointer: OrbaxCheckpointerProtocol[nnx.State[str | int, Array] | None] = ocp.PyTreeCheckpointer()
                    restored = checkpointer.restore(model_path)
                    if restored is not None and nnx is not None:
                        nnx.update(model, restored)
                except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
                    logger.warning("Failed to restore Orbax checkpoint from %s: %s", model_path, e)
                    if strict_checkpoint:
                        raise InferenceError(
                            f"Failed to restore Orbax checkpoint from {model_path}: {e!s}",
                            backend="jax",
                            model_name=model_name,
                            cause=e,
                        ) from e
            _MODEL_CACHE.put_model(model_name, model)

        def _model_forward(seq: Array, pos: Array) -> Array:
            """Call model with optional multimodal keyword arguments.

            Args:
                seq: Token ID sequence array.
                pos: Position index array.

            Returns:
                Model prediction logits array.

            """
            extra_kwargs: dict[str, ForwardArg] = {}
            if pixel_values is not None:
                extra_kwargs["pixel_values"] = pixel_values
            if audio_values is not None:
                extra_kwargs["audio_values"] = audio_values
            res: Array = call_jax_causal_lm(model, seq, pos, **extra_kwargs)  # pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]
            return res

        (output_ids, logprob_sum) = jax_beam_search(_model_forward, input_ids, beam_width, max_length, eos_token_id)
        sql_out = tokenizer.decode(output_ids[0].tolist())
        out_len = int(output_ids.shape[1])
        conf = normalize_logprob_confidence(logprob_sum, max(1, out_len - len(input_tokens)))
        return (sql_out, conf)

    wrapper_kwargs: dict[str, StatusValue] = {
        "raise_on_error": raise_on_error,
        "strict_checkpoint": strict_checkpoint,
    }
    return run_inference_wrapper("jax", model_name, prompt, beam_width, _execute, **wrapper_kwargs)
