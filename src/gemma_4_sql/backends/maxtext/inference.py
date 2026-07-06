"""Maxtext-specific inference logic."""

from __future__ import annotations

import logging
import operator
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.tokenization import SQLTokenizer

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)
jax = None
jnp = None
with catch_optional_imports():
    import jax
    import jax.numpy as jnp
Gemma4Model = None
with catch_optional_imports():
    from maxtext.models.gemma4 import Gemma4Model


def _beam_search_step(seq: jnp.ndarray, score: float, model_apply_fn: object, beam_width: int) -> list[tuple[jnp.ndarray, float]]:
    """Helper to process a single sequence and expand it into multiple beams.

    Args:
        seq: The seq.
        score: The float value for score.
        model_apply_fn: The model apply fn.
        beam_width: The number of beams for beam search.

    Returns:
        A tuple containing the results.
    """
    logits = model_apply_fn(seq)
    log_probs = jax.nn.log_softmax(logits, axis=-1)[0]
    top_indices = jnp.argsort(log_probs)[-beam_width:][::-1]
    top_probs = log_probs[top_indices]

    new_beams = []
    for i in range(beam_width):
        token = top_indices[i].reshape(1, 1)
        new_seq = jnp.concatenate([seq, token], axis=-1)
        new_score = score + top_probs[i].item()
        new_beams.append((new_seq, new_score))
    return new_beams


def maxtext_beam_search(model_apply_fn: object, input_ids: jnp.ndarray, beam_width: int, max_length: int, eos_token_id: int) -> tuple[jnp.ndarray, float]:
    """Maxtext native beam search implementation (XLA compiled via JIT).

    Returns:
        object: The resulting output from the operation.

    """
    beams = [(input_ids, 0.0)]
    for _ in range(max_length):
        new_beams = []
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


def _execute_generate(model_name: str, input_tokens: list[int], beam_width: int, max_length: int, eos_token_id: int, test_mode: bool, tokenizer: SQLTokenizer) -> tuple[str, str, float]:
    """Execute the generation logic.

    Args:
        model_name: The name of the model.
        input_tokens: The list of input tokens.
        beam_width: The beam width for search.
        max_length: The maximum length for generation.
        eos_token_id: The end-of-sequence token ID.
        test_mode: Whether to run in test mode.
        tokenizer: The tokenizer instance.

    Returns:
        A tuple of raw output text, clean SQL, and generation time.
    """
    logger.info("Generating with MaxText: %s", model_name)
    input_ids = jnp.array([input_tokens], dtype=jnp.int32)
    model = Gemma4Model(model_name)
    apply_fn = model.apply if hasattr(model, "apply") else model
    jitted_beam_search = jax.jit(maxtext_beam_search, static_argnums=(2, 3, 4))
    if not test_mode:
        (output_ids, logprob_sum) = jitted_beam_search(apply_fn, input_ids, beam_width, max_length, eos_token_id)
    else:
        (output_ids, logprob_sum) = maxtext_beam_search(apply_fn, input_ids, beam_width, max_length, eos_token_id)
    sql = tokenizer.decode(output_ids[0].tolist())
    out_len = len(output_ids[0]) if hasattr(output_ids[0], "__len__") else output_ids.shape[1]
    confidence_score = float(logprob_sum / max(1, out_len - len(input_tokens)))
    return "success", sql, confidence_score


def generate_sql(model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50, **kwargs: JSONValue) -> JSONDict:
    """Generate a SQL query from a natural language prompt using MaxText.

    Args:
    ----
        model_name: The name of the model to use.
        prompt: The natural language prompt.
        beam_width: Number of beams for search.
        max_length: Maximum number of tokens to generate.
        **kwargs: Extra arguments.

    Returns:
    -------
        A dictionary containing the generated SQL.

    """
    tokenizer = SQLTokenizer(model_name=None)
    input_tokens = tokenizer.encode(prompt)
    eos_token_id = tokenizer.vocab_size - 1
    confidence_score = 0.0
    if jax is None or jnp is None or Gemma4Model is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MaxText dependencies are missing.")
    try:
        status, sql, confidence_score = _execute_generate(model_name, input_tokens, beam_width, max_length, eos_token_id, bool(kwargs.get("test_mode")), tokenizer)
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("MaxText Generation Error: ")
        status = f"failed: {e!s}"
        sql = ""
    return {"backend": "maxtext", "model": model_name, "prompt": prompt, "sql": sql, "status": status, "beam_width": beam_width, "confidence_score": confidence_score}
