"""Maxtext-specific inference logic."""

from __future__ import annotations

import logging
import operator
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import TensorType

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)

jax = optional_import("jax")
jnp = optional_import("jax.numpy")
Gemma4Model = optional_attr("maxtext.models.gemma4", "Gemma4Model")


def _beam_search_step(seq: TensorType, score: float, model_apply_fn: Any, beam_width: int) -> list[tuple[Any, float]]:
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
    if hasattr(logits, "shape") and len(logits.shape) == 3:
        last_logits = logits[0, -1, :]
    elif hasattr(logits, "shape") and len(logits.shape) == 2:
        last_logits = logits[-1, :]
    else:
        last_logits = logits
    log_probs = jax.nn.log_softmax(last_logits, axis=-1)
    top_indices = jnp.argsort(log_probs)[-beam_width:][::-1]
    top_probs = log_probs[top_indices]

    new_beams: list[tuple[Any, float]] = []
    for i in range(beam_width):
        token = top_indices[i].reshape(1, 1)
        new_seq = jnp.concatenate([seq, token], axis=-1)
        prob_val = top_probs[i].item() if hasattr(top_probs[i], "item") else float(top_probs[i])
        new_score = score + prob_val
        new_beams.append((new_seq, new_score))
    return new_beams


def maxtext_beam_search(model_apply_fn: Any, input_ids: Any, beam_width: int, max_length: int, eos_token_id: int) -> tuple[Any, float]:
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
    beams: list[tuple[Any, float]] = [(input_ids, 0.0)]
    for _ in range(max_length):
        new_beams: list[tuple[Any, float]] = []
        for seq, score in beams:
            if seq[0, -1] == eos_token_id:
                new_beams.append((seq, score))
                continue

            expanded_beams = _beam_search_step(seq, score, model_apply_fn, beam_width)
            new_beams.extend(expanded_beams)

        new_beams.sort(key=operator.itemgetter(1), reverse=True)
        beams = new_beams[:beam_width]
        if all(seq[0, -1] == eos_token_id for (seq, _) in beams):
            break
    return (beams[0][0], beams[0][1])


def _execute_generate(model_name: str, input_tokens: list[int], beam_width: int, max_length: int, eos_token_id: int, tokenizer: SQLTokenizer) -> tuple[str, str, float]:
    """Execute the generation logic.

    Args:
        model_name: The name of the model.
        input_tokens: The list of input tokens.
        beam_width: The beam width for search.
        max_length: The maximum length for generation.
        eos_token_id: The end-of-sequence token ID.
        tokenizer: The tokenizer instance.

    Returns:
        A tuple of raw output text, clean SQL, and generation time.

    """
    logger.info("Generating with MaxText: %s", model_name)
    input_ids = jnp.array([input_tokens], dtype=jnp.int32)
    model = Gemma4Model(model_name)
    apply_fn = model.apply if hasattr(model, "apply") else model
    jitted_beam_search = jax.jit(maxtext_beam_search, static_argnums=(2, 3, 4))
    (output_ids, logprob_sum) = jitted_beam_search(apply_fn, input_ids, beam_width, max_length, eos_token_id)
    sql = tokenizer.decode(output_ids[0].tolist())
    out_len = len(output_ids[0]) if hasattr(output_ids[0], "__len__") else output_ids.shape[1]
    confidence_score = float(logprob_sum / max(1, out_len - len(input_tokens)))
    return "success", sql, confidence_score


def generate_sql(model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50, **kwargs: JSONValue) -> JSONDict:
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
    tokenizer = SQLTokenizer(model_name=None)
    input_tokens = tokenizer.encode(prompt)
    eos_token_id = tokenizer.vocab_size - 1
    confidence_score = 0.0
    if jax is None or jnp is None or Gemma4Model is None:
        raise DependencyMissingError("MaxText dependencies are missing.")
    try:
        status, sql, confidence_score = _execute_generate(model_name, input_tokens, beam_width, max_length, eos_token_id, tokenizer)
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("MaxText Generation Error: ")
        status = f"failed: {e!s}"
        sql = ""
    return {"backend": "maxtext", "model": model_name, "prompt": prompt, "sql": sql, "status": status, "beam_width": beam_width, "confidence_score": confidence_score}
