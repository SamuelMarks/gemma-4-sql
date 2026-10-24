"""MLX-specific inference logic and beam search."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError
from gemma_4_sql.type_hints import ModelType

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue

logger = logging.getLogger(__name__)

mx = optional_import("mlx.core")
load = optional_attr("mlx_lm", "load")
generate = optional_attr("mlx_lm", "generate")


def compute_confidence_score(log_probs: list[float] | float, num_tokens: int) -> float:
    """Compute length-normalized sequence confidence score from log probabilities.

    Mathematical formulation:
        confidence = exp( (1 / max(1, L)) * sum_{i=1}^L log P(t_i) )
    bounded within the range [0.0, 1.0].

    Args:
        log_probs: List of per-token log probabilities or cumulative log probability scalar.
        num_tokens: Total number of generated tokens L.

    Returns:
        Sequence confidence score between 0.0 and 1.0.

    """
    if num_tokens <= 0:
        return 0.0
    total_log_prob = sum(log_probs) if isinstance(log_probs, list) else float(log_probs)
    normalized_log_prob = total_log_prob / max(1, num_tokens)
    normalized_log_prob = min(normalized_log_prob, 0.0)
    return max(0.0, min(1.0, math.exp(normalized_log_prob)))


def mlx_beam_search(
    model: ModelType | None,
    tokenizer: Any,
    prompt: str,
    beam_width: int = 3,
    max_length: int = 50,
    eos_token_id: int | None = None,
) -> tuple[str, float]:
    """Perform beam search decoding using MLX.

    Maintains the top-k beam hypotheses scored by cumulative log-likelihood.
    At each autoregressive step, evaluates model logits for active beams,
    computes log-softmax probabilities, expands hypotheses, and prunes to top-k.

    Args:
        model: MLX model instance with callable forward pass.
        tokenizer: Tokenizer instance supporting encode and decode.
        prompt: Natural language input prompt string.
        beam_width: Number of beam hypotheses to maintain.
        max_length: Maximum number of new tokens to generate.
        eos_token_id: Optional end-of-sequence token ID.

    Returns:
        Tuple of (generated SQL query string, confidence score in [0.0, 1.0]).

    Raises:
        DependencyMissingError: If MLX dependencies are missing.
        InferenceError: If generation yields an empty sequence or model execution fails.

    """
    if model is None:
        raise InferenceError("Valid model instance is required for MLX beam search.")

    # Encode prompt
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(prompt)
        input_ids = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
    else:
        from gemma_4_sql.tokenization import SQLTokenizer

        input_ids = SQLTokenizer().encode(prompt)

    # Determine EOS ID
    if eos_token_id is not None:
        eos_id = eos_token_id
    elif tokenizer is not None and hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        eos_id = int(tokenizer.eos_token_id)
    else:
        eos_id = 1

    # Initialize beams: (generated_tokens, cumulative_log_prob, is_done)
    beams: list[tuple[list[int], float, bool]] = [([], 0.0, False)]

    for _ in range(max_length):
        candidates: list[tuple[list[int], float, bool]] = []

        for gen_tokens, cum_score, is_done in beams:
            if is_done:
                candidates.append((gen_tokens, cum_score, True))
                continue

            full_seq = input_ids + gen_tokens
            if mx is not None:
                tensor_in = mx.array([full_seq])
                out: Any = model(tensor_in)
            else:
                out = model(full_seq)

            next_logits: Any = None
            # Extract logits for the next token position
            if hasattr(out, "ndim") and out.ndim == 3:
                next_logits = out[0, -1, slice(None)]
            elif hasattr(out, "ndim") and out.ndim == 2:
                next_logits = out[-1, slice(None)]
            elif isinstance(out, (list, tuple)):
                next_logits = out[-1]
            else:
                next_logits = out

            if mx is not None and hasattr(next_logits, "shape"):
                log_probs = next_logits - mx.logsumexp(next_logits, axis=-1, keepdims=True)
                vocab_size = int(next_logits.shape[-1])
                k = min(beam_width, vocab_size)
                top_k_indices = mx.argsort(log_probs)[-k:].tolist()
                for tok_idx in top_k_indices:
                    idx = int(tok_idx)
                    log_prob_idx = log_probs[idx]
                    lp = float(log_prob_idx.item() if hasattr(log_prob_idx, "item") else log_prob_idx)
                    new_tokens = [*gen_tokens, idx]
                    new_cum = cum_score + lp
                    candidates.append((new_tokens, new_cum, idx == eos_id))
            else:
                candidates.append(([*gen_tokens, eos_id], cum_score, True))

        candidates.sort(key=lambda b: b[1] / max(1, len(b[0])), reverse=True)
        beams = candidates[:beam_width]
        if all(b[2] for b in beams):
            break

    best_tokens, best_score, _ = beams[0]
    if not best_tokens:
        raise InferenceError("MLX beam search yielded an empty sequence.")

    confidence = compute_confidence_score(best_score, len(best_tokens))

    # Strip EOS token if present
    output_tokens = best_tokens[:-1] if best_tokens and best_tokens[-1] == eos_id else best_tokens
    if not output_tokens:
        raise InferenceError("MLX beam search yielded only an EOS token.")

    if tokenizer is not None and hasattr(tokenizer, "decode"):
        sql = tokenizer.decode(output_tokens).strip()
    else:
        from gemma_4_sql.tokenization import SQLTokenizer

        sql = SQLTokenizer().decode(output_tokens).strip()

    if not sql:
        raise InferenceError("MLX beam search decoded into an empty SQL query string.")

    return (sql, confidence)


def generate_sql(
    model_name: str,
    prompt: str,
    beam_width: int = 3,
    max_length: int = 50,
    **kwargs: JSONValue,
) -> JSONDict:
    """Generate a SQL query from a natural language prompt using MLX.

    Args:
        model_name: The name of the target model.
        prompt: The input natural language prompt.
        beam_width: The number of beams for beam search.
        max_length: The maximum length of the sequence.
        **kwargs: Advanced generation parameters (e.g., eos_token_id).

    Returns:
        A dictionary containing the generated SQL and metadata.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.

    """
    if load is None:
        raise DependencyMissingError("MLX dependencies are missing.")

    confidence_score = 0.0
    sql = ""
    try:
        logger.info("Generating with MLX %s (beam_width=%d)", model_name, beam_width)
        loaded: Any = load(model_name)
        model: ModelType = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        tokenizer = loaded[1] if isinstance(loaded, (tuple, list)) and len(loaded) > 1 else None
        eos_id_arg = kwargs.get("eos_token_id")
        eos_id = int(eos_id_arg) if isinstance(eos_id_arg, (int, str, float)) else None
        sql, confidence_score = mlx_beam_search(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            beam_width=beam_width,
            max_length=max_length,
            eos_token_id=eos_id,
        )
        status = "success"
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("MLX generation failed: ")
        sql = ""
        confidence_score = 0.0
        status = f"failed: {e!s}"

    return {
        "backend": "mlx",
        "model": model_name,
        "prompt": prompt,
        "sql": sql,
        "status": status,
        "beam_width": beam_width,
        "confidence_score": confidence_score,
    }
