"""Common inference, LRU model caching, and beam search utilities across backends."""

from __future__ import annotations

import logging
import math
import operator
import threading
import weakref
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, TypeVar

from gemma_4_sql.exceptions import InferenceError, resolve_raise_on_error

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemma_4_sql.type_hints import InferenceResultDict

logger = logging.getLogger(__name__)
TModel = TypeVar("TModel")

_ACTIVE_LRU_CACHES: list[weakref.ref[BoundedLRUCache[Any]]] = []
_CACHE_REGISTRY_LOCK = threading.Lock()


class BoundedLRUCache(OrderedDict[str, TModel]):
    """Thread-safe LRU-bounded dictionary for caching initialized backend models or embeddings."""

    def __init__(self, maxsize: int = 4) -> None:
        """Initialize the bounded LRU cache.

        Args:
            maxsize: Maximum number of cached items to retain.

        """
        super().__init__()
        self.maxsize = maxsize
        self._lock = threading.Lock()
        with _CACHE_REGISTRY_LOCK:
            _ACTIVE_LRU_CACHES[:] = [r for r in _ACTIVE_LRU_CACHES if r() is not None]
            _ACTIVE_LRU_CACHES.append(weakref.ref(self))

    def get_model(self, key: str) -> TModel | None:
        """Retrieve a cached item and mark it as recently used.

        Args:
            key: Cache identifier key.

        Returns:
            Cached object or `None` if absent.

        """
        with self._lock:
            if key not in self:
                return None
            self.move_to_end(key)
            return self[key]

    def put_model(self, key: str, value: TModel) -> None:
        """Insert or update a cached item, evicting oldest entries while size exceeds `maxsize`.

        Args:
            key: Cache identifier key.
            value: Object to cache.

        """
        with self._lock:
            if key in self:
                self.move_to_end(key)
            super().__setitem__(key, value)
            while len(self) > max(0, self.maxsize):
                self.popitem(last=False)

    def clear_cache(self) -> None:
        """Thread-safely clear all entries from this LRU cache."""
        with self._lock:
            self.clear()


def clear_all_model_caches() -> None:
    """Clear all active `BoundedLRUCache` instances across all backends and SDK modules."""
    with _CACHE_REGISTRY_LOCK:
        alive_caches = [inst for ref in _ACTIVE_LRU_CACHES if (inst := ref()) is not None]
        _ACTIVE_LRU_CACHES[:] = [weakref.ref(c) for c in alive_caches]
    for cache in alive_caches:
        cache.clear_cache()


def normalize_logprob_confidence(log_probs: list[float] | float, num_tokens: int) -> float:
    """Compute length-normalized sequence confidence score from log probabilities in [0.0, 1.0].

    Args:
        log_probs: List of per-token log probabilities or cumulative log probability scalar.
        num_tokens: Total number of generated tokens.

    Returns:
        Sequence confidence score bounded in `[0.0, 1.0]`.

    """
    if num_tokens <= 0:
        return 0.0
    total_log_prob = sum(log_probs) if isinstance(log_probs, list) else float(log_probs)
    normalized_log_prob = min(total_log_prob / max(1, num_tokens), 0.0)
    return max(0.0, min(1.0, math.exp(normalized_log_prob)))


def run_array_beam_search(
    input_ids: Any,
    beam_width: int,
    max_length: int,
    eos_token_id: int,
    step_expand_fn: Callable[[Any, float, int], list[tuple[Any, float]]],
) -> tuple[Any, float]:
    """Run autoregressive beam search over 2D `(1, seq_len)` array sequences.

    Args:
        input_ids: Initial prompt token ID array of shape `(1, prompt_len)`.
        beam_width: Number of active beams to retain at each step.
        max_length: Maximum number of decoding steps.
        eos_token_id: End-of-sequence token ID.
        step_expand_fn: Callable `(seq, score, beam_width) -> list[(expanded_seq, new_score)]`.

    Returns:
        Tuple of `(best_sequence, cumulative_log_prob_score)`.

    """
    beams: list[tuple[Any, float]] = [(input_ids, 0.0)]
    for _ in range(max_length):
        new_beams: list[tuple[Any, float]] = []
        for seq, score in beams:
            if seq[0, -1] == eos_token_id:
                new_beams.append((seq, score))
                continue
            new_beams.extend(step_expand_fn(seq, score, beam_width))

        new_beams.sort(key=operator.itemgetter(1), reverse=True)
        beams = new_beams[:beam_width]
        if all(s[0, -1] == eos_token_id for (s, _) in beams):
            break
    return (beams[0][0], beams[0][1])


def compute_jax_step_probs(logits: Any, beam_width: int, jax_mod: Any, jnp_mod: Any) -> tuple[Any, Any]:
    """Compute top-k token indices and log probabilities from a JAX logits array.

    Args:
        logits: Logits tensor from model evaluation.
        beam_width: Number of top tokens to select.
        jax_mod: Loaded `jax` module.
        jnp_mod: Loaded `jax.numpy` module.

    Returns:
        Tuple of `(top_indices, top_probs)` arrays.

    """
    if hasattr(logits, "shape") and len(logits.shape) == 3:
        last_logits = logits[0, -1, :]
    elif hasattr(logits, "shape") and len(logits.shape) == 2:
        last_logits = logits[-1, :]
    else:
        last_logits = logits
    log_probs = jax_mod.nn.log_softmax(last_logits, axis=-1)
    top_indices = jnp_mod.argsort(log_probs)[-beam_width:][::-1]
    top_probs = log_probs[top_indices]
    return (top_indices, top_probs)


def expand_jax_beams(
    seq: Any,
    score: float,
    top_indices: Any,
    top_probs: Any,
    beam_width: int,
    jnp_mod: Any,
) -> list[tuple[Any, float]]:
    """Expand a sequence into `beam_width` candidate beams from top-k indices and log-probs.

    Args:
        seq: Current token ID sequence array of shape `(1, seq_len)`.
        score: Cumulative log-probability score of `seq`.
        top_indices: 1D array of top token IDs.
        top_probs: 1D array of top token log probabilities.
        beam_width: Number of beams to expand.
        jnp_mod: Loaded `jax.numpy` module.

    Returns:
        List of `(expanded_sequence, new_score)` tuples.

    """
    new_beams: list[tuple[Any, float]] = []
    for i in range(beam_width):
        token = top_indices[i].reshape(1, 1)
        new_seq = jnp_mod.concatenate([seq, token], axis=-1)
        prob_val = top_probs[i].item() if hasattr(top_probs[i], "item") else float(top_probs[i])
        new_score = score + prob_val
        new_beams.append((new_seq, new_score))
    return new_beams


def run_inference_wrapper(
    backend_name: str,
    model_name: str,
    prompt: str,
    beam_width: int,
    execute_fn: Callable[[], tuple[str, float]],
    **kwargs: object,
) -> InferenceResultDict:
    """Execute a backend inference pipeline with standardized error handling and result formatting.

    Args:
        backend_name: Target backend identifier ('jax', 'keras', 'maxtext', 'mlx', 'pytorch').
        model_name: Target model name or checkpoint path.
        prompt: Input prompt string (post-formatting if multimodal).
        beam_width: Number of beams used for decoding.
        execute_fn: Backend callable returning `(sql, confidence_score)`.
        **kwargs: Runtime keyword options (such as `raise_on_error=True` or `strict_checkpoint=True`).

    Returns:
        Structured `InferenceResultDict` containing generated SQL, confidence score, and status.

    Raises:
        InferenceError: If `execute_fn` raises an exception and `raise_on_error` (or `strict_checkpoint`) is enabled.

    """
    raise_on_error = resolve_raise_on_error(kwargs=dict(kwargs)) or bool(kwargs.get("strict_checkpoint"))
    sql = ""
    confidence_score = 0.0
    try:
        sql, confidence_score = execute_fn()
        status = "success"
    except (InferenceError, RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError, ImportError) as e:
        logger.exception("%s Generation Error: ", backend_name)
        if raise_on_error:
            if isinstance(e, InferenceError):
                raise
            raise InferenceError(
                f"{backend_name} generation failed: {e!s}",
                backend=backend_name,
                model_name=model_name,
                cause=e,
            ) from e
        status = f"failed: {e!s}"

    return {
        "backend": backend_name,
        "model": model_name,
        "prompt": prompt,
        "sql": sql,
        "status": status,
        "beam_width": beam_width,
        "confidence_score": float(confidence_score),
    }


__all__ = [
    "BoundedLRUCache",
    "clear_all_model_caches",
    "compute_jax_step_probs",
    "expand_jax_beams",
    "normalize_logprob_confidence",
    "run_array_beam_search",
    "run_inference_wrapper",
]
