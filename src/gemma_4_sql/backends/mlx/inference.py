"""MLX-specific inference logic and beam search."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Sequence
from typing import Protocol, SupportsFloat, runtime_checkable

from gemma_4_sql.backends.common_inference import (
    BoundedLRUCache,
    make_model_cache_key,
    normalize_logprob_confidence,
    run_inference_wrapper,
)
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError, StatusValue
from gemma_4_sql.type_hints import (
    InferenceResultDict,
    JSONValue,
    MlxArrayProtocol,
    MlxCausalLMProtocol,
    TokenizerProtocol,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _MlxTokenizerProtocol(Protocol):
    """Protocol for an MLX tokenizer supporting `.encode(text)`, `.decode(ids)`, and `.eos_token_id`."""

    @property
    def eos_token_id(self) -> int | None:
        """Provide the end-of-sequence token ID."""
        ...

    def encode(self, text: str, /) -> Sequence[int] | MlxArrayProtocol:
        """Encode `text` into token IDs."""
        ...

    def decode(self, token_ids: Sequence[int], /) -> str:
        """Decode `token_ids` into text."""
        ...


mx = optional_import("mlx.core")
_raw_load = optional_attr("mlx_lm", "load")
load: Callable[[str], tuple[MlxCausalLMProtocol, TokenizerProtocol | _MlxTokenizerProtocol | None] | MlxCausalLMProtocol] | None = (  # ty: ignore[invalid-assignment]
    _raw_load if callable(_raw_load) else None
)
generate = optional_attr("mlx_lm", "generate")

_MODEL_CACHE: BoundedLRUCache[tuple[MlxCausalLMProtocol, TokenizerProtocol | _MlxTokenizerProtocol | None]] = BoundedLRUCache(maxsize=4)
MODEL_CACHE = _MODEL_CACHE


def compute_confidence_score(log_probs: list[float] | float, num_tokens: int) -> float:
    """Compute length-normalized sequence confidence score from log probabilities.

    Mathematical formulation:
        `confidence = exp((1 / max(1, L)) * sum_{i=1}^L log P(t_i))`
    bounded within the range `[0.0, 1.0]`.

    Args:
        log_probs: List of per-token log probabilities or cumulative log probability scalar.
        num_tokens: Total number of generated tokens `L`.

    Returns:
        Sequence confidence score between `0.0` and `1.0`.

    """
    return normalize_logprob_confidence(log_probs, num_tokens)


def mlx_beam_search(
    model: MlxCausalLMProtocol | Callable[..., MlxArrayProtocol | SupportsFloat | Sequence[MlxArrayProtocol | SupportsFloat] | Sequence[int]] | None,
    tokenizer: TokenizerProtocol | _MlxTokenizerProtocol | None,
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
        tokenizer: Tokenizer instance supporting `encode` and `decode`.
        prompt: Natural language input prompt string.
        beam_width: Number of beam hypotheses to maintain.
        max_length: Maximum number of new tokens to generate.
        eos_token_id: Optional end-of-sequence token ID.

    Returns:
        Tuple of `(generated SQL query string, confidence score in [0.0, 1.0])`.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.
        InferenceError: If generation yields an empty sequence or model execution fails.

    """
    if model is None:
        raise InferenceError("Valid model instance is required for MLX beam search.")

    input_ids: list[int] | None = None
    with contextlib.suppress(AttributeError, TypeError):
        encoded = tokenizer.encode(prompt)  # pyright: ignore[reportOptionalMemberAccess]  # ty: ignore[unresolved-attribute]
        input_ids = [int(x) for x in encoded]
    if input_ids is None:
        from gemma_4_sql.tokenization import SQLTokenizer

        input_ids = SQLTokenizer().encode(prompt)

    eos_id = 1
    if eos_token_id is not None:
        eos_id = eos_token_id
    else:
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            eos_id = int(tokenizer.eos_token_id)  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue,reportArgumentType]  # ty: ignore[invalid-argument-type, unresolved-attribute]

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
                out = model(tensor_in)
            else:
                out = model(full_seq)

            if isinstance(out, MlxArrayProtocol):
                if out.ndim == 3:
                    next_logits: MlxArrayProtocol | None = out[0, -1, :]
                elif out.ndim == 2:
                    next_logits = out[-1, :]
                else:
                    next_logits = out
            else:
                last_item = out[-1]  # pyright: ignore[reportIndexIssue,reportArgumentType]  # ty: ignore[invalid-argument-type, not-subscriptable]
                next_logits = last_item if isinstance(last_item, MlxArrayProtocol) else None

            if mx is not None and next_logits is not None:
                log_probs: MlxArrayProtocol = next_logits - mx.logsumexp(next_logits, axis=-1, keepdims=True)
                vocab_size = int(next_logits.shape[-1])
                k = min(beam_width, vocab_size)
                top_k_indices = mx.argsort(log_probs)[-k:].tolist()
                for tok_idx in top_k_indices:
                    idx = int(tok_idx)
                    log_prob_idx = log_probs[idx]
                    lp = float(log_prob_idx.item())
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

    output_tokens = best_tokens[:-1] if best_tokens and best_tokens[-1] == eos_id else best_tokens
    if not output_tokens:
        raise InferenceError("MLX beam search yielded only an EOS token.")

    decoded_sql: str | None = None
    with contextlib.suppress(AttributeError, TypeError):
        decoded_sql = tokenizer.decode(output_tokens).strip()  # pyright: ignore[reportOptionalMemberAccess]  # ty: ignore[unresolved-attribute]
    if decoded_sql is None:
        from gemma_4_sql.tokenization import SQLTokenizer

        decoded_sql = SQLTokenizer().decode(output_tokens).strip()

    if not decoded_sql:
        raise InferenceError("MLX beam search decoded into an empty SQL query string.")

    return (decoded_sql, confidence)


def generate_sql(
    model_name: str,
    prompt: str,
    beam_width: int = 3,
    max_length: int = 50,
    **kwargs: JSONValue,
) -> InferenceResultDict:
    """Generate a SQL query from a natural language prompt using MLX.

    Args:
        model_name: The name of the target model.
        prompt: The input natural language prompt.
        beam_width: The number of beams for beam search.
        max_length: The maximum length of the sequence.
        **kwargs: Advanced generation parameters (e.g., `eos_token_id`).

    Returns:
        A dictionary containing the generated SQL and metadata.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.

    """
    if load is None:
        raise DependencyMissingError("MLX dependencies are missing.")

    if kwargs.get("image_path") is not None or kwargs.get("audio_path") is not None:
        from gemma_4_sql.backends.common_multimodal import format_multimodal_prompt

        mm_input = format_multimodal_prompt(
            prompt,
            has_image=kwargs.get("image_path") is not None,
            has_audio=kwargs.get("audio_path") is not None,
        )
        prompt = str(mm_input.get("prompt", prompt))

    def _execute() -> tuple[str, float]:
        """Load or retrieve cached MLX model and run beam search decoding."""
        assert load is not None
        logger.info("Generating with MLX %s (beam_width=%d)", model_name, beam_width)
        cache_key = make_model_cache_key(model_name, backend_name="mlx", factories=(load,))
        cached_pair = _MODEL_CACHE.get_model(cache_key)
        if cached_pair is not None:
            model, tokenizer = cached_pair
        else:
            loaded = load(model_name)
            model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
            tokenizer = loaded[1] if isinstance(loaded, (tuple, list)) and len(loaded) > 1 else None
            _MODEL_CACHE.put_model(cache_key, (model, tokenizer))
        eos_id_arg = kwargs.get("eos_token_id")
        eos_id = int(eos_id_arg) if isinstance(eos_id_arg, (int, str, float)) else None
        return mlx_beam_search(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            beam_width=beam_width,
            max_length=max_length,
            eos_token_id=eos_id,
        )

    wrapper_kwargs: dict[str, StatusValue] = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None}
    return run_inference_wrapper("mlx", model_name, prompt, beam_width, _execute, **wrapper_kwargs)
