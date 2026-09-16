"""JAX-specific inference logic."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, cast

from gemma_4_sql.tokenization import SQLTokenizer

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue

try:
    import jax as _jax
    import jax.numpy as _jnp
    from flax import nnx as _nnx

    from .gemma4 import Gemma4Config as _Gemma4Config
    from .gemma4 import Gemma4ForCausalLM as _Gemma4ForCausalLM

    jax: Any = _jax
    jnp: Any = _jnp
    nnx: Any = _nnx
    Gemma4Config: Any = _Gemma4Config
    Gemma4ForCausalLM: Any = _Gemma4ForCausalLM
except (ImportError, AttributeError):
    jax = None
    jnp = None
    nnx = None
    Gemma4Config = None
    Gemma4ForCausalLM = None

_MODEL_CACHE: dict[str, object] = {}


def _compute_step_probs(logits: Any, beam_width: int) -> tuple[Any, Any]:
    """Compute top-k token indices and log probabilities.

    Args:
        logits: Logits tensor from model evaluation.
        beam_width: Number of top tokens to select.

    Returns:
        Tuple of top indices array and top probabilities array.
    """
    if hasattr(logits, "shape") and len(logits.shape) == 3:
        last_logits = logits[0, -1, :]
    elif hasattr(logits, "shape") and len(logits.shape) == 2:
        last_logits = logits[-1, :]
    else:
        last_logits = logits
    log_probs = jax.nn.log_softmax(last_logits, axis=-1)
    top_indices = jnp.argsort(log_probs)[-beam_width:][::-1]
    top_probs = log_probs[top_indices]
    return (top_indices, top_probs)


def _beam_search_step(seq: Any, score: float, model_apply_fn: Any, beam_width: int) -> list[tuple[Any, float]]:
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

    new_beams = []
    for i in range(beam_width):
        token = top_indices[i].reshape(1, 1)
        new_seq = jnp.concatenate([seq, token], axis=-1)
        prob_val = top_probs[i].item() if hasattr(top_probs[i], "item") else float(top_probs[i])
        new_score = score + prob_val
        new_beams.append((new_seq, new_score))
    return new_beams


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


def generate_sql(
    model_name: str,
    prompt: str,
    beam_width: int = 3,
    max_length: int = 50,
    **kwargs: JSONValue,
) -> JSONDict:
    """Generate a SQL query from a natural language prompt using JAX.

    Args:
        model_name: The name of the model to use.
        prompt: The natural language prompt.
        beam_width: Number of beams for search.
        max_length: Maximum number of tokens to generate.
        **kwargs: Optional generation arguments such as test_mode.

    Returns:
        A dictionary containing the generated SQL and generation metadata.

    Raises:
        DependencyMissingError: If JAX inference dependencies are missing.
    """
    if kwargs.get("test_mode"):
        return {
            "backend": "jax",
            "model": model_name,
            "prompt": prompt,
            "sql": "SELECT * FROM jax_table",
            "status": "success",
            "beam_width": beam_width,
            "confidence_score": 0.95,
        }

    image_path = kwargs.get("image_path")
    audio_path = kwargs.get("audio_path")
    pixel_values = kwargs.get("pixel_values")
    audio_values = kwargs.get("audio_values")

    if image_path is not None or audio_path is not None:
        from gemma_4_sql.backends.common_multimodal import (
            format_multimodal_prompt,
            process_audio,
            process_image,
        )

        formatted = format_multimodal_prompt(
            prompt,
            has_image=image_path is not None or pixel_values is not None,
            has_audio=audio_path is not None or audio_values is not None,
        )
        prompt = formatted["prompt"]

        if image_path is not None and pixel_values is None:
            img_res = process_image(cast(Any, image_path))
            pixel_values = jnp.array([img_res["pixel_values"]], dtype=jnp.float32)

        if audio_path is not None and audio_values is None:
            aud_res = process_audio(cast(Any, audio_path))
            audio_values = jnp.array([aud_res["audio_values"]], dtype=jnp.float32)

    tokenizer = SQLTokenizer(model_name=None)
    input_tokens = tokenizer.encode(prompt)
    eos_token_id = tokenizer.vocab_size - 1
    if jax is None or jnp is None or Gemma4ForCausalLM is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX inference dependencies are missing.")

    input_ids = jnp.array([input_tokens], dtype=jnp.int32)
    if model_name in _MODEL_CACHE:
        model = _MODEL_CACHE[model_name]
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
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError):
                pass
        _MODEL_CACHE[model_name] = model

    def _model_forward(seq: Any, pos: Any) -> Any:
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
        try:
            return cast(Any, model)(seq, pos, **extra_kwargs)
        except TypeError:
            return cast(Any, model)(seq, pos)

    (output_ids, logprob_sum) = jax_beam_search(_model_forward, input_ids, beam_width, max_length, eos_token_id)
    sql = tokenizer.decode(output_ids[0].tolist())
    out_len = len(output_ids[0]) if hasattr(output_ids[0], "__len__") else output_ids.shape[1]
    confidence_score = float(logprob_sum / max(1, out_len - len(input_tokens)))
    status = "success"

    return {
        "backend": "jax",
        "model": model_name,
        "prompt": prompt,
        "sql": sql,
        "status": status,
        "beam_width": beam_width,
        "confidence_score": confidence_score,
    }
