"""JAX-specific inference logic with cached beam step compilation and KV-cached incremental decoding."""

from __future__ import annotations

import logging
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
from flax import nnx
from jax import Array

from gemma_4_sql.backends._jax_common import (
    JaxJitBeamModuleProtocol,
    JaxModuleBeamProtocol,
    JaxNumpyBeamProtocol,
    build_jax_beam_step_fn,
    call_jax_causal_lm,
    compute_jax_step_probs,
    run_jax_beam_search_step,
)
from gemma_4_sql.backends.common_inference import (
    BoundedLRUCache,
    normalize_logprob_confidence,
    run_array_beam_search,
    run_inference_wrapper,
)
from gemma_4_sql.backends.common_multimodal import prepare_multimodal_inputs
from gemma_4_sql.backends.jax._protocols import is_jnp_missing
from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.jax.gemma4.cache import Cache, LayerCache, init_cache
from gemma_4_sql.backends.jax.gemma4.config import ModelConfig, resolve_gemma4_config
from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM
from gemma_4_sql.backends.jax.gemma4.modeling import forward as forward_fn
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError, StatusValue
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import (
    BackendKwargValue,
    ForwardArg,
    InferenceResultDict,
    JaxArrayProtocol,
    JaxCausalLMProtocol,
    NDArrayProtocol,
    OrbaxCheckpointerProtocol,
    extract_str_kwarg,
)

logger = logging.getLogger(__name__)

Gemma4Config = ModelConfig
_Gemma4ForCausalLMCls = Gemma4ForCausalLM

MAX_MODEL_CACHE_SIZE = 4
_MODEL_CACHE: BoundedLRUCache[JaxCausalLMProtocol | Gemma4ForCausalLM | Callable[..., Array]] = BoundedLRUCache(
    maxsize=MAX_MODEL_CACHE_SIZE
)
MODEL_CACHE = _MODEL_CACHE

type _CompiledBeamStepFn = Callable[[Array, float, Array, int], list[tuple[Array, float]]]


@dataclass
class _BeamStepCacheSlot:
    """Mutable cache slot storing the compiled XLA beam-search step expander.

    Attributes:
        jit_ref: Last seen `jax.jit` callable reference (or `None` when `jax.jit` is absent).
        step_fn: Cached compiled beam-step expander callable.

    """

    jit_ref: Callable[..., Callable[..., tuple[Array, Array]]] | None = None
    step_fn: _CompiledBeamStepFn | None = None


_BEAM_STEP_SLOT = _BeamStepCacheSlot()


def _get_cached_beam_step_fn() -> _CompiledBeamStepFn:
    """Return a module-cached compiled beam-search step expander.

    Rebuilds only if `jax.jit` identity changes (for example during test monkeypatching),
    avoiding per-request XLA re-tracing across `jax_beam_search` calls.

    Returns:
        Compiled `(seq, score, logits, beam_width) -> list[(expanded_seq, new_score)]` callable.

    """
    jax_beam_mod: JaxModuleBeamProtocol[Array] = jax
    jnp_beam_mod: JaxNumpyBeamProtocol[Array] = jnp
    current_jit = jax.jit if isinstance(jax, JaxJitBeamModuleProtocol) else None
    if _BEAM_STEP_SLOT.step_fn is not None and current_jit is _BEAM_STEP_SLOT.jit_ref:
        return _BEAM_STEP_SLOT.step_fn
    step_fn = build_jax_beam_step_fn(jax_beam_mod, jnp_beam_mod)
    _BEAM_STEP_SLOT.jit_ref = current_jit
    _BEAM_STEP_SLOT.step_fn = step_fn
    return step_fn


def _clone_kv_cache(cache: Cache) -> Cache:
    """Create an independent shallow tensor copy of `cache` so sibling beams do not alias `LayerCache` buffers.

    Args:
        cache: Source list of per-layer `LayerCache` modules.

    Returns:
        Cloned `Cache` list with independent `nnx.Cache` wrappers per layer.

    """
    cloned: Cache = []
    for lc in cache:
        shape = (
            int(lc.k_cache[...].shape[0]),
            int(lc.k_cache[...].shape[1]),
            int(lc.k_cache[...].shape[2]),
            int(lc.k_cache[...].shape[3]),
        )
        new_lc = LayerCache(shape, lc.k_cache[...].dtype)
        new_lc.k_cache[...] = lc.k_cache[...]
        new_lc.v_cache[...] = lc.v_cache[...]
        new_lc.cur_ind[...] = lc.cur_ind[...]
        cloned.append(new_lc)
    return cloned


def _compute_step_probs(logits: Array, beam_width: int) -> tuple[Array, Array]:
    """Compute top-k token indices and log probabilities.

    Args:
        logits: Logits tensor from model evaluation.
        beam_width: Number of top tokens to select.

    Returns:
        Tuple of `(top_indices, top_probs)` arrays.

    """
    return compute_jax_step_probs(logits, beam_width, jax, jnp)


def _beam_search_step(
    seq: Array,
    score: float,
    model_apply_fn: Callable[[Array, Array], Array],
    beam_width: int,
) -> list[tuple[Array, float]]:
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
    return run_jax_beam_search_step(seq, score, logits, beam_width, jax, jnp)


def jax_beam_search(
    model_apply_fn: Callable[[Array, Array], Array],
    input_ids: Array,
    beam_width: int,
    max_length: int,
    eos_token_id: int,
) -> tuple[Array, float]:
    """Run JAX native beam search decoding over an arbitrary `model_apply_fn`.

    Args:
        model_apply_fn: The model's forward pass function.
        input_ids: The initial input token IDs.
        beam_width: The number of beams to maintain.
        max_length: The maximum generation length.
        eos_token_id: The end-of-sequence token ID.

    Returns:
        The sequence of token IDs representing the best beam and its score.

    """
    compiled_step = _get_cached_beam_step_fn()

    def _step_expand(seq: Array, score: float, bw: int) -> list[tuple[Array, float]]:
        """Evaluate model logits for `seq` and expand beams using the module-cached JIT step function.

        Args:
            seq: Current token ID sequence array.
            score: Cumulative log-probability score of `seq`.
            bw: Beam width integer.

        Returns:
            List of `(expanded_sequence, new_score)` tuples.

        """
        positions = jnp.arange(seq.shape[1])[None, :]
        logits = model_apply_fn(seq, positions)
        return compiled_step(seq, score, logits, bw)

    return run_array_beam_search(
        input_ids,
        beam_width,
        max_length,
        eos_token_id,
        _step_expand,
    )


def jax_kv_cached_beam_search(
    model: Gemma4ForCausalLM,
    input_ids: Array,
    beam_width: int,
    max_length: int,
    eos_token_id: int,
    *,
    pixel_values: Array | None = None,
    audio_values: Array | None = None,
) -> tuple[Array, float]:
    """Run incremental `O(T)` KV-cached beam search on `Gemma4ForCausalLM` via `forward_fn`.

    Prefills the prompt once into `LayerCache` and feeds single-token slices `(1, 1)` on each
    subsequent decoding step rather than recomputing attention over the full prefix (`O(T^2)`).

    Args:
        model: `Gemma4ForCausalLM` instance.
        input_ids: Prompt token IDs of shape `(1, prompt_len)`.
        beam_width: Number of beams to maintain.
        max_length: Maximum number of new tokens to decode.
        eos_token_id: End-of-sequence token ID.
        pixel_values: Optional vision input tensor for prompt prefill.
        audio_values: Optional audio feature tensor for prompt prefill.

    Returns:
        Tuple of `(best_sequence_ids, best_cumulative_logprob)`.

    """
    prompt_len = int(input_ids.shape[1])
    if max_length <= 0 or (prompt_len > 0 and int(input_ids[0, -1]) == eos_token_id):
        return (input_ids, 0.0)

    compiled_step = _get_cached_beam_step_fn()
    total_max_len = prompt_len + max_length + 1
    prefill_cache = init_cache(model.config, batch_size=1, max_seq_len=total_max_len)
    prefill_pos = jnp.arange(prompt_len, dtype=jnp.int32)[None, :]
    next_logits, prefill_cache = forward_fn(
        model,
        prefill_cache,
        input_ids,
        prefill_pos,
        pixel_values=pixel_values,
        input_features=audio_values,
    )
    initial_expanded = compiled_step(input_ids, 0.0, next_logits, beam_width)
    beams: list[tuple[Array, float, Cache]] = [
        (cand_seq, cand_score, _clone_kv_cache(prefill_cache)) for cand_seq, cand_score in initial_expanded
    ]

    for _ in range(1, max_length):
        if all(int(seq[0, -1]) == eos_token_id for seq, _, _ in beams):
            break
        candidates: list[tuple[Array, float, Cache]] = []
        for seq, score, beam_cache in beams:
            if int(seq[0, -1]) == eos_token_id:
                candidates.append((seq, score, beam_cache))
                continue
            step_cache = _clone_kv_cache(beam_cache)
            last_token = seq[:, -1:]
            last_pos = jnp.array([[int(seq.shape[1]) - 1]], dtype=jnp.int32)
            step_logits, step_cache = forward_fn(model, step_cache, last_token, last_pos)
            for new_seq, new_score in compiled_step(seq, score, step_logits, beam_width):
                candidates.append((new_seq, new_score, _clone_kv_cache(step_cache)))
        candidates.sort(key=operator.itemgetter(1), reverse=True)
        beams = candidates[:beam_width]

    best_seq, best_score, _ = beams[0]
    return (best_seq, best_score)


def generate_sql(
    model_name: str,
    prompt: str,
    beam_width: int = 3,
    max_length: int = 50,
    **kwargs: BackendKwargValue | Array,
) -> InferenceResultDict:
    """Generate a SQL query from a natural language prompt using JAX.

    Args:
        model_name: The name of the model to use.
        prompt: The natural language prompt.
        beam_width: Number of beams for search.
        max_length: Maximum number of tokens to generate.
        **kwargs: Optional generation arguments.

    Returns:
        A dictionary containing the generated SQL and generation metadata.

    Raises:
        DependencyMissingError: If JAX inference dependencies are missing.

    """
    if jax is None or is_jnp_missing(jnp) or not callable(Gemma4ForCausalLM):
        raise DependencyMissingError("JAX inference dependencies are missing.")

    def _to_jax_tensor(
        arr: NDArrayProtocol
        | JaxArrayProtocol
        | Sequence[float]
        | Sequence[Sequence[float]]
        | Sequence[Sequence[Sequence[float]]],
    ) -> Array:
        """Convert feature array into a batched float32 JAX array.

        Args:
            arr: Preprocessed image or audio feature array.

        Returns:
            Batched float32 JAX `Array`.

        """
        return jnp.array([arr], dtype=jnp.float32)

    def _passthrough_jax(val: BackendKwargValue | None) -> Array | None:
        """Return `val` when it is already a JAX `Array`, otherwise `None`.

        Args:
            val: Candidate multimodal keyword argument value.

        Returns:
            JAX `Array` instance or `None`.

        """
        return val if isinstance(val, Array) else None

    (prompt, pixel_values, audio_values) = prepare_multimodal_inputs(
        prompt,
        dict(kwargs),
        _to_jax_tensor,
        _passthrough_jax,
    )

    json_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool)) or v is None}
    tokenizer_name = extract_str_kwarg(json_kwargs, "tokenizer_name")
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    input_tokens = tokenizer.encode(prompt)
    eos_token_id = tokenizer.eos_token_id
    strict_checkpoint = bool(kwargs.get("strict_checkpoint"))
    raise_on_error = bool(kwargs.get("raise_on_error"))

    def _execute() -> tuple[str, float]:
        """Run JAX model loading, optional Orbax restore, and KV-cached or generic beam search decoding.

        Returns:
            Tuple of `(decoded_sql_string, confidence_score)`.

        Raises:
            InferenceError: If `strict_checkpoint=True` and Orbax checkpoint loading fails.

        """
        input_ids = jnp.array([input_tokens], dtype=jnp.int32)
        cached_model = _MODEL_CACHE.get_model(model_name)
        if cached_model is not None:
            model = cached_model
        else:
            assert nnx is not None
            rngs = nnx.Rngs(0)
            cfg = resolve_gemma4_config(Gemma4Config, model_name)
            model = Gemma4ForCausalLM(cfg, rngs=rngs)
            model_path = Path(model_name)
            if model_path.exists():
                try:
                    import orbax.checkpoint as ocp

                    from gemma_4_sql.type_hints import PyTreeNode

                    ckpt_factory: Callable[[], OrbaxCheckpointerProtocol[Mapping[str, PyTreeNode] | None]] = vars(ocp)[
                        "PyTreeCheckpointer"
                    ]
                    checkpointer = ckpt_factory()
                    restored = checkpointer.restore(model_path)
                    if restored is not None and nnx is not None:
                        update_fn: Callable[[Gemma4ForCausalLM, Mapping[str, PyTreeNode]], None] = vars(nnx)["update"]
                        update_fn(model, restored)
                except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
                    logger.warning("Failed to restore Orbax checkpoint from %s: %s", model_path, e)
                    if strict_checkpoint:
                        raise InferenceError(
                            f"Failed to restore Orbax checkpoint from {model_path}: {e!s}",
                            backend="jax",
                            model_name=model_name,
                            cause=e,
                        ) from e
            _MODEL_CACHE.put_model(model_name, model)

        if isinstance(model, _Gemma4ForCausalLMCls):
            (output_ids, logprob_sum) = jax_kv_cached_beam_search(
                model,
                input_ids,
                beam_width,
                max_length,
                eos_token_id,
                pixel_values=pixel_values,
                audio_values=audio_values,
            )
        else:

            def _model_forward(seq: Array, pos: Array) -> Array:
                """Call generic model callable with optional multimodal keyword arguments.

                Args:
                    seq: Token ID sequence array.
                    pos: Position index array.

                Returns:
                    Model prediction logits array.

                """
                extra_kwargs: dict[str, ForwardArg] = {}
                if pixel_values is not None:
                    extra_kwargs["pixel_values"] = pixel_values
                if audio_values is not None:
                    extra_kwargs["audio_values"] = audio_values
                raw_out = call_jax_causal_lm(model, seq, pos, **extra_kwargs)
                first_out = raw_out[0] if isinstance(raw_out, tuple) else raw_out
                return first_out if isinstance(first_out, Array) else jnp.asarray(first_out)

            (output_ids, logprob_sum) = jax_beam_search(_model_forward, input_ids, beam_width, max_length, eos_token_id)

        sql_out = tokenizer.decode(output_ids[0].tolist())
        out_len = int(output_ids.shape[1])
        conf = normalize_logprob_confidence(logprob_sum, max(1, out_len - len(input_tokens)))
        return (sql_out, conf)

    wrapper_kwargs: dict[str, StatusValue] = {
        "raise_on_error": raise_on_error,
        "strict_checkpoint": strict_checkpoint,
    }
    return run_inference_wrapper("jax", model_name, prompt, beam_width, _execute, **wrapper_kwargs)


__all__ = [
    "MODEL_CACHE",
    "Gemma4Config",
    "_beam_search_step",
    "_compute_step_probs",
    "forward_fn",
    "generate_sql",
    "jax_beam_search",
    "jax_kv_cached_beam_search",
]
