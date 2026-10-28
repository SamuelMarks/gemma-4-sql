"""Common inference, LRU model caching, and beam search utilities across backends."""

from __future__ import annotations

import inspect
import logging
import math
import operator
import threading
import types
import weakref
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, Self, SupportsFloat, TypeVar, runtime_checkable

from gemma_4_sql.exceptions import (
    OPERATIONAL_EXCEPTIONS,
    ConfigurationError,
    DependencyMissingError,
    ExecutionStatus,
    InferenceError,
    StatusValue,
    resolve_raise_on_error,
)
from gemma_4_sql.type_hints import (
    ForwardArg,
    InferenceResultDict,
    JSONValue,
    ModelConfigProtocol,
)

logger = logging.getLogger(__name__)
TModel = TypeVar("TModel")


@runtime_checkable
class CacheableRoutineProtocol(Protocol):
    """Protocol for an active `BoundedLRUCache` supporting `clear_cache()`."""

    def clear_cache(self) -> None:
        """Clear all entries in the cache."""
        ...


@runtime_checkable
class CacheFingerprintProtocol(Protocol):
    """Protocol for configuration objects providing an explicit `cache_fingerprint()` method."""

    def cache_fingerprint(self) -> str:
        """Return a deterministic string fingerprint for model caching."""
        ...


@runtime_checkable
class IndexableSequence2DProtocol(Protocol):
    """Protocol for a 2D sequence array supporting `seq[row, col]` indexing."""

    def __getitem__(self, item: tuple[int, int], /) -> SupportsFloat:
        """Return element at `(row, col)`."""
        ...


@runtime_checkable
class BeamSequenceArrayProtocol(SupportsFloat, Protocol):
    """Protocol for 2D token ID sequence arrays indexed by `(row, col)`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the array dimensions."""
        ...

    def __getitem__(
        self,
        item: int | slice | types.EllipsisType | tuple[int | slice | types.EllipsisType, ...],
        /,
    ) -> Self:
        """Index into the sequence array."""
        ...

    def reshape(self, *shape: int) -> Self:
        """Reshape the array."""
        ...


@runtime_checkable
class ScalarItemExtractorProtocol(Protocol):
    """Protocol for a scalar tensor element supporting `.item()`."""

    def item(self) -> float | int | bool:
        """Extract a Python float/int scalar."""
        ...


@runtime_checkable
class JaxNNBeamNamespaceProtocol[ArrayT](Protocol):
    """Protocol for `jax.nn` providing `log_softmax`."""

    def log_softmax(self, x: ArrayT, axis: int = -1) -> ArrayT:
        """Compute log-softmax along `axis`."""
        ...


@runtime_checkable
class JaxModuleBeamProtocol[ArrayT](Protocol):
    """Protocol for the `jax` module exposing `.nn.log_softmax`."""

    @property
    def nn(self) -> JaxNNBeamNamespaceProtocol[ArrayT]:
        """Provide the `jax.nn` submodule."""
        ...


@runtime_checkable
class JaxNumpyBeamProtocol[ArrayT](Protocol):
    """Protocol for `jax.numpy` providing `argsort` and `concatenate`."""

    def argsort(self, a: ArrayT, axis: int = -1) -> ArrayT:
        """Return the indices that would sort an array."""
        ...

    def concatenate(self, arrays: Sequence[ArrayT], axis: int = -1) -> ArrayT:
        """Join a sequence of arrays along an existing axis."""
        ...


type CacheConfigInput = Mapping[str, JSONValue] | CacheFingerprintProtocol | ModelConfigProtocol | str | int | float | bool | None

_ACTIVE_LRU_CACHES: list[weakref.ref[CacheableRoutineProtocol]] = []
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


def _fingerprint_config(config: CacheConfigInput) -> str:
    """Compute a deterministic fingerprint string for a model configuration object.

    Args:
        config: Configuration dataclass, mapping, or protocol instance.

    Returns:
        Deterministic string fingerprint.

    """
    if isinstance(config, CacheFingerprintProtocol):
        return config.cache_fingerprint()
    if isinstance(config, Mapping):
        items = sorted((str(k), repr(v)) for k, v in config.items())
        return ",".join(f"{k}={v}" for k, v in items)
    return repr(config)


@runtime_checkable
class _CacheFactoryItemProtocol(Protocol):
    """Protocol satisfied by any factory callable, class, or object passed to `make_model_cache_key`."""

    def __repr__(self) -> str:
        """Return string representation."""
        ...


def make_model_cache_key(
    model_name: str,
    *,
    backend_name: str = "",
    dtype: str = "",
    quant_method: str = "",
    adapter_path: str = "",
    config: CacheConfigInput = None,
    factories: tuple[_CacheFactoryItemProtocol | str | int | None, ...] = (),
) -> str:
    """Build a canonical cache key scoped by backend, model name, dtype, quantization method, adapter, and factories.

    Args:
        model_name: Target model name or path.
        backend_name: Optional backend identifier.
        dtype: Optional dtype string.
        quant_method: Optional quantization method string.
        adapter_path: Optional LoRA or PEFT adapter path.
        config: Optional configuration instance.
        factories: Optional tuple of factory callables or classes to scope cache across test overrides.

    Returns:
        Canonical string cache key.

    """
    parts = [p for p in (backend_name, model_name, dtype, quant_method, adapter_path) if p]
    if config is not None:
        parts.append(_fingerprint_config(config))
    parts.extend(str(id(item)) for item in factories if item is not None)
    return ":".join(parts) if len(parts) > 1 else model_name


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


def run_array_beam_search[SeqT: IndexableSequence2DProtocol](
    input_ids: SeqT,
    beam_width: int,
    max_length: int,
    eos_token_id: int,
    step_expand_fn: Callable[[SeqT, float, int], list[tuple[SeqT, float]]],
) -> tuple[SeqT, float]:
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
    beams: list[tuple[SeqT, float]] = [(input_ids, 0.0)]
    for _ in range(max_length):
        new_beams: list[tuple[SeqT, float]] = []
        for seq, score in beams:
            if int(float(seq[0, -1])) == eos_token_id:
                new_beams.append((seq, score))
                continue
            new_beams.extend(step_expand_fn(seq, score, beam_width))

        new_beams.sort(key=operator.itemgetter(1), reverse=True)
        beams = new_beams[:beam_width]
        if all(int(float(s[0, -1])) == eos_token_id for (s, _) in beams):
            break
    return (beams[0][0], beams[0][1])


def compute_jax_step_probs[ArrayT: BeamSequenceArrayProtocol](
    logits: ArrayT,
    beam_width: int,
    jax_mod: JaxModuleBeamProtocol[ArrayT],
    jnp_mod: JaxNumpyBeamProtocol[ArrayT],
) -> tuple[ArrayT, ArrayT]:
    """Compute top-k token indices and log probabilities from a JAX logits array.

    Args:
        logits: Logits tensor from model evaluation.
        beam_width: Number of top tokens to select.
        jax_mod: Loaded `jax` module.
        jnp_mod: Loaded `jax.numpy` module.

    Returns:
        Tuple of `(top_indices, top_probs)` arrays.

    """
    if len(logits.shape) == 3:
        last_logits: ArrayT = logits[0, -1, :]
    elif len(logits.shape) == 2:
        last_logits = logits[-1, :]
    else:
        last_logits = logits
    log_probs = jax_mod.nn.log_softmax(last_logits, axis=-1)
    sorted_indices: ArrayT = jnp_mod.argsort(log_probs)
    top_tail: ArrayT = sorted_indices[-beam_width:]
    top_indices: ArrayT = top_tail[::-1]
    top_probs: ArrayT = log_probs[top_indices]  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
    return (top_indices, top_probs)


def expand_jax_beams[ArrayT: BeamSequenceArrayProtocol](
    seq: ArrayT,
    score: float,
    top_indices: ArrayT,
    top_probs: ArrayT,
    beam_width: int,
    jnp_mod: JaxNumpyBeamProtocol[ArrayT],
) -> list[tuple[ArrayT, float]]:
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
    new_beams: list[tuple[ArrayT, float]] = []
    for i in range(beam_width):
        tok_elem: ArrayT = top_indices[i]
        token = tok_elem.reshape(1, 1)
        new_seq = jnp_mod.concatenate([seq, token], axis=-1)
        raw_prob = top_probs[i]
        prob_val = float(raw_prob.item()) if isinstance(raw_prob, ScalarItemExtractorProtocol) else float(raw_prob)
        new_score = score + prob_val
        new_beams.append((new_seq, new_score))
    return new_beams


def run_inference_wrapper(
    backend_name: str,
    model_name: str,
    prompt: str,
    beam_width: int,
    execute_fn: Callable[[], tuple[str, float]],
    **kwargs: StatusValue,
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
    error_message: str | None = None
    try:
        sql, confidence_score = execute_fn()
        status = ExecutionStatus.SUCCESS.value
        status_code = ExecutionStatus.SUCCESS.value
    except (DependencyMissingError, ConfigurationError):
        raise
    except OPERATIONAL_EXCEPTIONS as e:
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
        error_message = str(e)
        status = f"failed: {error_message}"
        status_code = ExecutionStatus.FAILED.value

    result: InferenceResultDict = {
        "backend": backend_name,
        "model": model_name,
        "prompt": prompt,
        "sql": sql,
        "status": status,
        "status_code": status_code,
        "beam_width": beam_width,
        "confidence_score": float(confidence_score),
    }
    if error_message is not None:
        result["error_message"] = error_message
    return result


def call_jax_causal_lm[OutT](
    model: Callable[..., OutT],
    inputs: ForwardArg,
    positions: ForwardArg = None,
    **extra_kwargs: ForwardArg,
) -> OutT:
    """Invoke a JAX causal language model with signature-aware positional and keyword argument forwarding.

    Inspects `inspect.signature(model)` instead of catching broad `TypeError` during execution,
    ensuring internal `TypeError` bugs inside `model.__call__` are never masked.

    Args:
        model: Callable JAX/Flax NNX model or function.
        inputs: Input token ID array.
        positions: Optional token position index array.
        **extra_kwargs: Optional multimodal or forward keyword arguments (e.g., `pixel_values`, `audio_values`).

    Returns:
        Model output logits array.

    """
    sig = inspect.signature(model)
    params = list(sig.parameters.values())
    has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    pos_params = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    accepts_positions = has_var_pos or len(pos_params) >= 2 or "positions" in sig.parameters
    accepted_kwargs = extra_kwargs if has_var_kw else {k: v for k, v in extra_kwargs.items() if k in sig.parameters}

    if positions is not None and accepts_positions:
        return model(inputs, positions, **accepted_kwargs)
    return model(inputs, **accepted_kwargs)


__all__ = [
    "BeamSequenceArrayProtocol",
    "BoundedLRUCache",
    "CacheConfigInput",
    "CacheFingerprintProtocol",
    "CacheableRoutineProtocol",
    "JaxModuleBeamProtocol",
    "JaxNNBeamNamespaceProtocol",
    "JaxNumpyBeamProtocol",
    "ScalarItemExtractorProtocol",
    "call_jax_causal_lm",
    "clear_all_model_caches",
    "compute_jax_step_probs",
    "expand_jax_beams",
    "make_model_cache_key",
    "normalize_logprob_confidence",
    "run_array_beam_search",
    "run_inference_wrapper",
]
