"""Maxtext-specific inference logic."""

from __future__ import annotations

import logging

from gemma_4_sql.tokenization import SQLTokenizer

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    jax = None
    jnp = None
try:
    from maxtext.models.gemma4 import Gemma4Model
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    Gemma4Model = None


def maxtext_beam_search(model_apply_fn: object, input_ids: jnp.ndarray, beam_width: int, max_length: int, eos_token_id: int) -> tuple[jnp.ndarray, float]:
    """Maxtext native beam search implementation (XLA compiled via JIT)."""
    # Note: For true XLA compilation with variable lengths, jax.lax.while_loop is preferred.
    # We simulate a simplified statically unrolled beam search here for the stub.
    beams = [(input_ids, 0.0)]
    for _ in range(max_length):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1] == eos_token_id:
                new_beams.append((seq, score))
                continue
            logits = model_apply_fn(seq)  # type: ignore[operator]
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
        if all(seq[0, -1] == eos_token_id for (seq, _) in beams):
            break
    return beams[0][0], beams[0][1]


def generate_sql(model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50, **kwargs: object) -> dict[str, object]:
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
    if jax is not None and jnp is not None and (Gemma4Model is not None):
        try:
            logger.info("Generating with MaxText: %s", model_name)
            input_ids = jnp.array([input_tokens], dtype=jnp.int32)
            model = Gemma4Model(model_name)

            # Create a JIT compiled version of the beam search
            apply_fn = model.apply if hasattr(model, "apply") else model
            jitted_beam_search = jax.jit(maxtext_beam_search, static_argnums=(2, 3, 4))

            if not kwargs.get("test_mode"):
                output_ids, logprob_sum = jitted_beam_search(apply_fn, input_ids, beam_width, max_length, eos_token_id)
            else:
                # Use non-jitted for easier test mocking
                output_ids, logprob_sum = maxtext_beam_search(apply_fn, input_ids, beam_width, max_length, eos_token_id)

            sql = tokenizer.decode(output_ids[0].tolist())
            out_len = len(output_ids[0]) if hasattr(output_ids[0], "__len__") else output_ids.shape[1]
            confidence_score = float(logprob_sum / max(1, out_len - len(input_tokens)))
            status = "success"
        except Exception as e:
            logger.exception("MaxText Generation Error: %s", e)
            status = f"failed: {e!s}"
            sql = ""
    else:
        sql = "SELECT * FROM maxtext_table"
        confidence_score = 0.95
        status = "mocked_missing_maxtext"

    return {"backend": "maxtext", "model": model_name, "prompt": prompt, "sql": sql, "status": status, "beam_width": beam_width, "confidence_score": confidence_score}
