"""MaxText-specific inference logic with `(model, params)` caching and Orbax checkpoint loading."""

from __future__ import annotations

import logging
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self, final, runtime_checkable

from gemma_4_sql._types_models import MaxTextModelConstructorProtocol
from gemma_4_sql.backends._jax_common import (
    BeamSequenceArrayProtocol,
    JaxModuleBeamProtocol,
    JaxNumpyBeamProtocol,
    build_jax_beam_step_fn,
    run_jax_beam_search_step,
)
from gemma_4_sql.backends.common_inference import (
    BoundedLRUCache,
    normalize_logprob_confidence,
    run_array_beam_search,
    run_inference_wrapper,
)
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.backends.maxtext._protocols import (
    JaxPRNGHolderProtocol,
    JnpZerosAndArrayProtocol,
    OrbaxCheckpointModuleProtocol,
    OrbaxManagerProtocol,
)
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import (
    InferenceResultDict,
    JaxArrayProtocol,
    JSONValue,
    MaxTextCausalLMProtocol,
    PyTreeNode,
    extract_status_kwargs,
    extract_str_kwarg,
)

logger = logging.getLogger(__name__)

jax: JaxPRNGHolderProtocol | None = optional_import("jax", JaxPRNGHolderProtocol)
jnp: JnpZerosAndArrayProtocol | None = optional_import("jax.numpy", JnpZerosAndArrayProtocol)
ocp: OrbaxCheckpointModuleProtocol | None = optional_import("orbax.checkpoint", OrbaxCheckpointModuleProtocol)


@runtime_checkable
class _MaxTextInferenceFactoryProtocol(Protocol):
    """Protocol for `Gemma4Model` returning either a stateless `MaxTextCausalLMProtocol` or direct callable."""

    def __call__(
        self,
        model_name: str,
        /,
    ) -> MaxTextCausalLMProtocol | Callable[[JaxArrayProtocol], JaxArrayProtocol]:
        """Instantiate a MaxText model or inference callable.

        Args:
            model_name: Target model name.

        Returns:
            Stateless `MaxTextCausalLMProtocol` or direct logits callable.

        """
        ...


Gemma4Model: _MaxTextInferenceFactoryProtocol | MaxTextModelConstructorProtocol | None = import_optional_symbol(
    "maxtext.models.gemma4",
    "Gemma4Model",
    MaxTextModelConstructorProtocol,
)


@final
@dataclass(frozen=True)
class CachedMaxTextModelPair:
    """Cached pair of stateless `MaxTextCausalLMProtocol` and initialized parameter PyTree.

    Attributes:
        model: Stateless MaxText model instance.
        params: Parameter PyTree dictionary.

    """

    model: MaxTextCausalLMProtocol
    params: dict[str, PyTreeNode]


type CachedMaxTextEntry = CachedMaxTextModelPair | Callable[[JaxArrayProtocol], JaxArrayProtocol]

MAX_MODEL_CACHE_SIZE = 4
_MODEL_CACHE: BoundedLRUCache[CachedMaxTextEntry] = BoundedLRUCache(maxsize=MAX_MODEL_CACHE_SIZE)
_PARAMS_ONLY_CACHE: BoundedLRUCache[dict[str, PyTreeNode]] = BoundedLRUCache(maxsize=MAX_MODEL_CACHE_SIZE)


@runtime_checkable
class _OrbaxManagerProtocol(Protocol):
    """Protocol for an Orbax `CheckpointManager` context manager."""

    def __enter__(self) -> Self:
        """Enter context manager.

        Returns:
            Active checkpoint manager.

        """
        ...

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: types.TracebackType | None
    ) -> bool | None:
        """Exit context manager.

        Args:
            exc_type: Exception class if raised.
            exc: Exception instance if raised.
            tb: Traceback object if raised.

        Returns:
            Optional suppression flag.

        """
        ...

    def latest_step(self) -> int | None:
        """Return latest persisted step integer or `None`.

        Returns:
            Latest step index or `None`.

        """
        ...

    def restore(self, step: int) -> Mapping[str, PyTreeNode]:
        """Restore checkpoint PyTree at `step`.

        Args:
            step: Checkpoint step integer.

        Returns:
            Restored parameter PyTree dictionary.

        """
        ...


def _resolve_orbax_dir(candidate_path: str | Path) -> Path | None:
    """Locate a valid Orbax checkpoint directory at `candidate_path` or `candidate_path / 'maxtext_orbax_ckpt'`.

    Args:
        candidate_path: Filesystem path to inspect.

    Returns:
        Resolved Orbax checkpoint directory `Path`, or `None` if no checkpoint exists.

    """
    base = Path(candidate_path)
    sub = base / "maxtext_orbax_ckpt"
    for target in (sub, base):
        if target.is_dir() and any(child.is_dir() and child.name.isdigit() for child in target.iterdir()):
            return target.resolve()
    return None


def load_maxtext_orbax_params(checkpoint_path: str | Path) -> dict[str, PyTreeNode] | None:
    """Restore MaxText parameters from an Orbax checkpoint directory if present.

    Args:
        checkpoint_path: Checkpoint directory or parent export directory.

    Returns:
        Restored parameter PyTree dictionary, or `None` when not an Orbax checkpoint directory.

    """
    resolved_dir = _resolve_orbax_dir(checkpoint_path)
    if resolved_dir is None or ocp is None:
        return None
    mngr: OrbaxManagerProtocol = ocp.CheckpointManager(resolved_dir, ocp.PyTreeCheckpointer())
    with mngr as active_mngr:
        step = active_mngr.latest_step()
        if step is None:
            return None
        restored = dict(active_mngr.restore(step))
    nested_params = restored.get("params")
    if isinstance(nested_params, Mapping) and set(restored.keys()) <= {"params", "opt_state"}:
        return {str(k): v for k, v in nested_params.items()}
    return {str(k): v for k, v in restored.items()}


def cache_maxtext_params(
    model_name: str,
    params: Mapping[str, PyTreeNode],
    model: MaxTextCausalLMProtocol | None = None,
) -> None:
    """Store `(model, params)` or standalone `params` in the MaxText LRU caches.

    Args:
        model_name: Model cache identifier.
        params: Parameter PyTree dictionary to cache.
        model: Optional stateless `MaxTextCausalLMProtocol` instance.

    """
    params_copy: dict[str, PyTreeNode] = {str(k): v for k, v in params.items()}
    _PARAMS_ONLY_CACHE.put_model(model_name, params_copy)
    existing = _MODEL_CACHE.get_model(model_name)
    if model is not None:
        _MODEL_CACHE.put_model(model_name, CachedMaxTextModelPair(model=model, params=params_copy))
    elif isinstance(existing, CachedMaxTextModelPair):
        _MODEL_CACHE.put_model(model_name, CachedMaxTextModelPair(model=existing.model, params=params_copy))


def get_cached_maxtext_params(model_name: str) -> dict[str, PyTreeNode] | None:
    """Retrieve cached MaxText parameters for `model_name` if available.

    Args:
        model_name: Model cache identifier.

    Returns:
        Cached parameter PyTree dictionary, or `None` if absent.

    """
    entry = _MODEL_CACHE.get_model(model_name)
    if isinstance(entry, CachedMaxTextModelPair):
        return dict(entry.params)
    standalone = _PARAMS_ONLY_CACHE.get_model(model_name)
    return dict(standalone) if standalone is not None else None


def _get_or_create_maxtext_model(
    model_name: str,
    sample_input: JaxArrayProtocol | None = None,
    checkpoint_path: str | None = None,
) -> CachedMaxTextEntry:
    """Retrieve a cached `(model, params)` pair (or callable model) or initialize/restore and cache it.

    Args:
        model_name: Target model name or checkpoint path.
        sample_input: Optional sample token ID array used only on initial parameter creation.
        checkpoint_path: Optional explicit Orbax checkpoint directory to restore parameters from.

    Returns:
        Cached `CachedMaxTextModelPair` or direct callable model.

    """
    cache_key = f"{model_name}::{checkpoint_path}" if checkpoint_path else model_name
    cached = _MODEL_CACHE.get_model(cache_key)
    if cached is not None:
        return cached
    assert Gemma4Model is not None
    assert jax is not None
    assert jnp is not None
    model = Gemma4Model(model_name)
    if isinstance(model, MaxTextCausalLMProtocol):
        restored = load_maxtext_orbax_params(checkpoint_path) if checkpoint_path else None
        if restored is None:
            restored = load_maxtext_orbax_params(model_name)
        if restored is None and checkpoint_path is None:
            restored = _PARAMS_ONLY_CACHE.get_model(model_name)
        if restored is not None:
            params: dict[str, PyTreeNode] = dict(restored)
        else:
            init_input: JaxArrayProtocol = (
                sample_input if sample_input is not None else jnp.zeros((1, 4), dtype=jnp.int32)
            )
            rng = jax.random.PRNGKey(0)
            params = {str(k): v for k, v in dict(model.init(rng, init_input)).items()}
        entry: CachedMaxTextEntry = CachedMaxTextModelPair(model=model, params=params)
        _MODEL_CACHE.put_model(cache_key, entry)
        if checkpoint_path is None:
            _PARAMS_ONLY_CACHE.put_model(model_name, params)
        return entry
    _MODEL_CACHE.put_model(cache_key, model)
    return model


def _beam_search_step[ArrayT: BeamSequenceArrayProtocol](
    seq: ArrayT,
    score: float,
    model_apply_fn: Callable[[ArrayT], ArrayT],
    beam_width: int,
) -> list[tuple[ArrayT, float]]:
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
    assert isinstance(jax, JaxModuleBeamProtocol)
    assert isinstance(jnp, JaxNumpyBeamProtocol)
    return run_jax_beam_search_step(seq, score, logits, beam_width, jax, jnp)


def maxtext_beam_search[ArrayT: BeamSequenceArrayProtocol](
    model_apply_fn: Callable[[ArrayT], ArrayT],
    input_ids: ArrayT,
    beam_width: int,
    max_length: int,
    eos_token_id: int,
) -> tuple[ArrayT, float]:
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
    assert isinstance(jax, JaxModuleBeamProtocol)
    assert isinstance(jnp, JaxNumpyBeamProtocol)
    compiled_step: Callable[[ArrayT, float, ArrayT, int], list[tuple[ArrayT, float]]] = build_jax_beam_step_fn(jax, jnp)

    def _step_expand(seq: ArrayT, score: float, bw: int) -> list[tuple[ArrayT, float]]:
        """Evaluate model logits for `seq` and expand beams using the hoisted JIT step function.

        Args:
            seq: Current partial beam sequence array.
            score: Current cumulative log-probability score.
            bw: Active beam width.

        Returns:
            Expanded `(sequence, score)` beam candidates.

        """
        logits = model_apply_fn(seq)
        return compiled_step(seq, score, logits, bw)

    return run_array_beam_search(
        input_ids,
        beam_width,
        max_length,
        eos_token_id,
        _step_expand,
    )


def _execute_generate(
    model_name: str,
    input_tokens: list[int],
    beam_width: int,
    max_length: int,
    eos_token_id: int,
    tokenizer: SQLTokenizer,
    checkpoint_path: str | None = None,
) -> tuple[str, str, float]:
    """Execute MaxText beam search generation using cached or checkpoint-restored parameters.

    Args:
        model_name: The name or checkpoint path of the model.
        input_tokens: Encoded prompt token IDs.
        beam_width: Beam width for search.
        max_length: Maximum number of generated tokens.
        eos_token_id: End-of-sequence token ID.
        tokenizer: Tokenizer instance for decoding output tokens.
        checkpoint_path: Optional Orbax checkpoint directory to restore parameters from.

    Returns:
        A tuple of `(status, clean_sql, confidence_score)`.

    """
    from gemma_4_sql.backends.maxtext.peft import apply_with_lora

    assert jax is not None
    assert jnp is not None
    logger.info("Generating with MaxText: %s", model_name)
    input_ids: JaxArrayProtocol = jnp.array([input_tokens], dtype=jnp.int32)
    cached_entry = _get_or_create_maxtext_model(model_name, sample_input=input_ids, checkpoint_path=checkpoint_path)
    if isinstance(cached_entry, CachedMaxTextModelPair):
        model = cached_entry.model
        params = cached_entry.params

        def _apply_with_params(seq_in: JaxArrayProtocol) -> JaxArrayProtocol:
            """Apply stateless MaxText model with cached/restored (and optional LoRA/AQT) parameters.

            Args:
                seq_in: Input token sequence array.

            Returns:
                Output vocabulary logits array.

            """
            return apply_with_lora(model, params, seq_in)

        apply_fn: Callable[[JaxArrayProtocol], JaxArrayProtocol] = _apply_with_params
    else:
        apply_fn = cached_entry
    (output_ids, logprob_sum) = maxtext_beam_search(apply_fn, input_ids, beam_width, max_length, eos_token_id)
    raw_list = output_ids[0].tolist()
    if isinstance(raw_list, list):
        tok_list = [int(t) for t in raw_list if isinstance(t, (int, float, bool))]
    else:
        scalar_tok = int(raw_list.real) if isinstance(raw_list, complex) else int(raw_list)
        tok_list = [scalar_tok]
    sql = tokenizer.decode(tok_list)
    out_len = int(output_ids.shape[1])
    confidence_score = normalize_logprob_confidence(logprob_sum, max(1, out_len - len(input_tokens)))
    return "success", sql, confidence_score


def generate_sql(
    model_name: str,
    prompt: str,
    beam_width: int = 3,
    max_length: int = 50,
    **kwargs: JSONValue,
) -> InferenceResultDict:
    """Generate a SQL query from a natural language prompt using MaxText.

    Args:
        model_name: The name or checkpoint path of the model to use.
        prompt: The natural language prompt.
        beam_width: Number of beams for search.
        max_length: Maximum number of tokens to generate.
        **kwargs: Extra arguments (including optional `checkpoint_path` and `tokenizer_name`).

    Returns:
        A dictionary containing the generated SQL and confidence score.

    Raises:
        DependencyMissingError: If MaxText dependencies are missing.

    """
    if jax is None or jnp is None or Gemma4Model is None:
        raise DependencyMissingError("MaxText dependencies are missing.")

    tokenizer_name = extract_str_kwarg(kwargs, "tokenizer_name")
    checkpoint_path = extract_str_kwarg(kwargs, "checkpoint_path")
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    input_tokens = tokenizer.encode(prompt)
    eos_token_id = tokenizer.eos_token_id

    def _execute() -> tuple[str, float]:
        """Run MaxText beam search generation and return `(sql, confidence_score)`.

        Returns:
            Tuple of `(sql, confidence_score)`.

        """
        _status, sql, confidence_score = _execute_generate(
            model_name,
            input_tokens,
            beam_width,
            max_length,
            eos_token_id,
            tokenizer,
            checkpoint_path=checkpoint_path,
        )
        return (sql, confidence_score)

    wrapper_kwargs = extract_status_kwargs(kwargs)
    return run_inference_wrapper("maxtext", model_name, prompt, beam_width, _execute, **wrapper_kwargs)


__all__ = [
    "_beam_search_step",
    "_execute_generate",
    "cache_maxtext_params",
    "generate_sql",
    "get_cached_maxtext_params",
    "load_maxtext_orbax_params",
    "maxtext_beam_search",
]
