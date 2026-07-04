# Copyright 2024
"""JAX-specific inference logic."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.tokenization import SQLTokenizer

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
jax = None
jnp = None
with catch_optional_imports():
    import jax
    import jax.numpy as jnp
Gemma4ForCausalLM = None
Gemma4Config = None
nnx = None
with catch_optional_imports():
    from flax import nnx

    from .gemma4 import Gemma4Config, Gemma4ForCausalLM


def _beam_search_step(seq: jnp.ndarray, score: float, model_apply_fn: object, beam_width: int) -> list[tuple[jnp.ndarray, float]]:
    """Helper to process a single sequence and expand it into multiple beams.

    Returns:
        object: Description of return.

    """
    positions = jnp.arange(seq.shape[1])[None, :]
    logits = model_apply_fn(seq, positions)
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


def jax_beam_search(model_apply_fn: object, input_ids: jnp.ndarray, beam_width: int, max_length: int, eos_token_id: int) -> tuple[jnp.ndarray, float]:
    """JAX native beam search implementation.

    Args:
    ----
        model_apply_fn: The model's forward pass function.
        input_ids: The initial input token IDs.
        beam_width: The number of beams to maintain.
        max_length: The maximum generation length.
        eos_token_id: The end-of-sequence token ID.

    Returns:
    -------
        The sequence of token IDs representing the best beam.

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
        if all(s[0, -1] == eos_token_id for s, _ in beams):
            break
    return (beams[0][0], beams[0][1])


def generate_sql(model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50) -> JSONDict:
    """Generate a SQL query from a natural language prompt using JAX.

    Args:
    ----
        model_name: The name of the model to use.
        prompt: The natural language prompt.
        beam_width: Number of beams for search.
        max_length: Maximum number of tokens to generate.

    Returns:
    -------
        A dictionary containing the generated SQL.

    """
    tokenizer = SQLTokenizer(model_name=None)
    input_tokens = tokenizer.encode(prompt)
    eos_token_id = tokenizer.vocab_size - 1
    confidence_score = 0.0
    if jax is not None and jnp is not None and (Gemma4ForCausalLM is not None):
        input_ids = jnp.array([input_tokens], dtype=jnp.int32)
        model = Gemma4ForCausalLM(Gemma4Config.gemma4_e2b(), rngs=nnx.Rngs(0))
        (output_ids, logprob_sum) = jax_beam_search(model, input_ids, beam_width, max_length, eos_token_id)
        sql = tokenizer.decode(output_ids[0].tolist())
        out_len = len(output_ids[0]) if hasattr(output_ids[0], "__len__") else output_ids.shape[1]
        confidence_score = float(logprob_sum / max(1, out_len - len(input_tokens)))
        status = "success"
    else:
        sql = "SELECT * FROM jax_table"
        confidence_score = 0.95
        status = "mocked_missing_jax"
    return {"backend": "jax", "model": model_name, "prompt": prompt, "sql": sql, "status": status, "beam_width": beam_width, "confidence_score": confidence_score}
