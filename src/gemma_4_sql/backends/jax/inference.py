"""JAX-specific inference logic."""

from __future__ import annotations

import logging
from typing import Any

from gemma_4_sql.backends.common_inference import (
    BoundedLRUCache,
    call_jax_causal_lm,
    compute_jax_step_probs,
    expand_jax_beams,
    run_array_beam_search,
    run_inference_wrapper,
)
from gemma_4_sql.backends.common_multimodal import prepare_multimodal_inputs
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import InferenceError
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import InferenceResultDict, TensorType

logger = logging.getLogger(__name__)

jax = optional_import("jax")
jnp = optional_import("jax.numpy")
nnx = optional_import("flax.nnx")
Gemma4Config = optional_attr("gemma_4_sql.backends.jax.gemma4", "Gemma4Config")
Gemma4ForCausalLM = optional_attr("gemma_4_sql.backends.jax.gemma4", "Gemma4ForCausalLM")

MAX_MODEL_CACHE_SIZE = 4
_BoundedModelCache = BoundedLRUCache
_MODEL_CACHE: BoundedLRUCache[object] = BoundedLRUCache(maxsize=MAX_MODEL_CACHE_SIZE)


def _compute_step_probs(logits: TensorType, beam_width: int) -> tuple[Any, Any]:
    """Compute top-k token indices and log probabilities.

    Args:
        logits: Logits tensor from model evaluation.
        beam_width: Number of top tokens to select.

    Returns:
        Tuple of top indices array and top probabilities array.

    """
    return compute_jax_step_probs(logits, beam_width, jax, jnp)


def _beam_search_step(seq: TensorType, score: float, model_apply_fn: Any, beam_width: int) -> list[tuple[Any, float]]:
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
    if jax is not None and hasattr(jax, "jit"):
        step_fn = jax.jit(_compute_step_probs, static_argnums=(1,))
        (top_indices, top_probs) = step_fn(logits, beam_width)
    else:
        (top_indices, top_probs) = _compute_step_probs(logits, beam_width)
    return expand_jax_beams(seq, score, top_indices, top_probs, beam_width, jnp)


def jax_beam_search(model_apply_fn: Any, input_ids: Any, beam_width: int, max_length: int, eos_token_id: int) -> tuple[Any, float]:
    """JAX native beam search implementation.

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
    **kwargs: object,
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
    if jax is None or jnp is None or Gemma4ForCausalLM is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX inference dependencies are missing.")

    def _to_jax_tensor(arr: Any) -> Any:
        """Convert feature array into a batched float32 JAX array."""
        return jnp.array([arr], dtype=jnp.float32)

    (prompt, pixel_values, audio_values) = prepare_multimodal_inputs(
        prompt,
        dict(kwargs),
        _to_jax_tensor,
    )

    tokenizer = SQLTokenizer(model_name=None)
    input_tokens = tokenizer.encode(prompt)
    eos_token_id = tokenizer.vocab_size - 1

    def _execute() -> tuple[str, float]:
        """Run JAX model loading, optional Orbax restore, and beam search decoding."""
        input_ids = jnp.array([input_tokens], dtype=jnp.int32)
        cached_model = _MODEL_CACHE.get_model(model_name)
        if cached_model is not None:
            model = cached_model
        else:
            rngs = nnx.Rngs(0) if nnx is not None and hasattr(nnx, "Rngs") else None
            model = Gemma4ForCausalLM(Gemma4Config.gemma4_e2b(), rngs=rngs)
            from pathlib import Path

            model_path = Path(model_name)
            if model_path.exists():
                try:
                    import orbax.checkpoint as ocp

                    checkpointer = ocp.PyTreeCheckpointer()
                    restored = checkpointer.restore(model_path)
                    if restored is not None and nnx is not None and hasattr(nnx, "update"):
                        nnx.update(model, restored)
                except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
                    logger.warning("Failed to restore Orbax checkpoint from %s: %s", model_path, e)
                    if kwargs.get("strict_checkpoint"):
                        raise InferenceError(
                            f"Failed to restore Orbax checkpoint from {model_path}: {e!s}",
                            backend="jax",
                            model_name=model_name,
                            cause=e,
                        ) from e
            _MODEL_CACHE.put_model(model_name, model)

        def _model_forward(seq: TensorType, pos: Any) -> Any:
            """Call model with optional multimodal keyword arguments.

            Args:
                seq: Token ID sequence array.
                pos: Position index array.

            Returns:
                Model prediction logits array.

            """
            extra_kwargs: dict[str, Any] = {}
            if pixel_values is not None:
                extra_kwargs["pixel_values"] = pixel_values
            if audio_values is not None:
                extra_kwargs["audio_values"] = audio_values
            return call_jax_causal_lm(model, seq, pos, **extra_kwargs)

        (output_ids, logprob_sum) = jax_beam_search(_model_forward, input_ids, beam_width, max_length, eos_token_id)
        sql_out = tokenizer.decode(output_ids[0].tolist())
        out_len = len(output_ids[0]) if hasattr(output_ids[0], "__len__") else output_ids.shape[1]
        conf = float(logprob_sum / max(1, out_len - len(input_tokens)))
        return (sql_out, conf)

    return run_inference_wrapper("jax", model_name, prompt, beam_width, _execute, **kwargs)
