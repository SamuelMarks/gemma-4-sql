"""
JAX-specific inference logic.
"""

from __future__ import annotations

from typing import Any

from gemma_4_sql.tokenization import SQLTokenizer

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None

try:
    from bonsai.models.gemma4 import Gemma4Config, Gemma4ForCausalLM
    from flax import nnx  # pragma: no cover
except Exception:
    Gemma4ForCausalLM = None
    Gemma4Config = None
    nnx = None


def jax_beam_search(
    model_apply_fn: Any,
    input_ids: jnp.ndarray,
    beam_width: int,
    max_length: int,
    eos_token_id: int,
) -> jnp.ndarray:
    """
    JAX native beam search implementation.
    """
    beams = [(input_ids, 0.0)]

    for _ in range(max_length):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1] == eos_token_id:
                new_beams.append((seq, score))
                continue

            logits = model_apply_fn(seq)
            log_probs = jax.nn.log_softmax(logits, axis=-1)[0]

            top_indices = jnp.argsort(log_probs)[-beam_width:][::-1]
            top_probs = log_probs[top_indices]

            for i in range(beam_width):
                token = top_indices[i].reshape(1, 1)
                new_seq = jnp.concatenate([seq, token], axis=-1)
                new_score = score + top_probs[i].item()
                new_beams.append((new_seq, new_score))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

        if all(seq[0, -1] == eos_token_id for seq, _ in beams):
            break

    return beams[0][0]


def generate_sql(
    model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50
) -> dict[str, Any]:
    """
    Generates a SQL query from a natural language prompt using JAX.

    Args:
        model_name: The name of the model to use.
        prompt: The natural language prompt.
        beam_width: Number of beams for search.
        max_length: Maximum number of tokens to generate.

    Returns:
        A dictionary containing the generated SQL.
    """
    tokenizer = SQLTokenizer(model_name=None)
    input_tokens = tokenizer.encode(prompt)
    eos_token_id = tokenizer.vocab_size - 1

    if jax is not None and jnp is not None and Gemma4ForCausalLM is not None:
        input_ids = jnp.array([input_tokens], dtype=jnp.int32)
        model = Gemma4ForCausalLM(Gemma4Config.gemma4_e2b(), rngs=nnx.Rngs(0))

        output_ids = jax_beam_search(
            model,
            input_ids,
            beam_width,
            max_length,
            eos_token_id,
        )
        sql = tokenizer.decode(output_ids[0].tolist())
        status = "success"
    else:
        sql = "SELECT * FROM jax_table"
        status = "mocked_missing_jax"

    return {
        "backend": "jax",
        "model": model_name,
        "prompt": prompt,
        "sql": sql,
        "status": status,
        "beam_width": beam_width,
    }
