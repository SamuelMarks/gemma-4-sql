"""Shared JAX and Flax NNX utilities for `jax` and `maxtext` backends."""

from __future__ import annotations

import inspect
import logging
import os
import types
import weakref
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, Self, SupportsFloat, runtime_checkable

from gemma_4_sql._types_tensors import ScalarItemExtractorProtocol, extract_scalar_float
from gemma_4_sql.backends.common_benchmark import normalize_hardware_family
from gemma_4_sql.backends.lazy_loader import ModuleNamespaceProtocol
from gemma_4_sql.exceptions import HardwareUnavailableError
from gemma_4_sql.type_hints import ForwardArg, JaxShardingProtocol

logger = logging.getLogger(__name__)
_jax_dist_init_state: list[bool] = [False]


@runtime_checkable
class JaxSliceableTensorProtocol(SupportsFloat, Protocol):
    """Protocol for JAX-like arrays supporting slicing, arithmetic, and masking."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide tensor dimensions.

        Returns:
            Tuple of dimension sizes.

        """
        ...

    def __getitem__(
        self,
        item: int | slice | types.EllipsisType | Self | tuple[int | slice | types.EllipsisType, ...],
        /,
    ) -> Self:
        """Index or slice the tensor.

        Args:
            item: Index, slice, ellipsis, or tuple of slices.

        Returns:
            Sliced tensor view.

        """
        ...

    def __mul__(self, other: Self | float, /) -> Self:
        """Multiply tensor elementwise.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Elementwise product tensor.

        """
        ...

    def __rmul__(self, other: Self | float, /) -> Self:
        """Right-multiply tensor elementwise.

        Args:
            other: Left-hand tensor or scalar operand.

        Returns:
            Elementwise product tensor.

        """
        ...


@runtime_checkable
class JaxNNLogSoftmaxProtocol[ArrayT](Protocol):
    """Protocol for `jax.nn` providing `log_softmax`."""

    def log_softmax(self, x: ArrayT, axis: int = -1) -> ArrayT:
        """Compute log-softmax along `axis`.

        Args:
            x: Input logits array.
            axis: Normalization axis index.

        Returns:
            Log-probabilities array.

        """
        ...


@runtime_checkable
class JaxNumpyLogpOpsProtocol[ArrayT](Protocol):
    """Protocol for `jax.numpy` operations required by `gather_jax_sequence_logps`."""

    def expand_dims(self, a: ArrayT, axis: int) -> ArrayT:
        """Insert a singleton axis.

        Args:
            a: Input array.
            axis: Axis index to insert.

        Returns:
            Expanded array.

        """
        ...

    def take_along_axis(self, arr: ArrayT, indices: ArrayT, axis: int) -> ArrayT:
        """Gather values along `axis` at `indices`.

        Args:
            arr: Source array.
            indices: Integer index array.
            axis: Axis along which to gather.

        Returns:
            Gathered array.

        """
        ...

    def squeeze(self, a: ArrayT, axis: int) -> ArrayT:
        """Remove a singleton axis.

        Args:
            a: Input array.
            axis: Singleton axis index to remove.

        Returns:
            Squeezed array.

        """
        ...

    def not_equal(self, x1: ArrayT, x2: int, /) -> ArrayT:
        """Compute elementwise inequality mask `x1 != x2`.

        Args:
            x1: Input array.
            x2: Integer token ID to mask out.

        Returns:
            Boolean inequality mask array.

        """
        ...

    def sum(self, a: ArrayT, axis: int = -1) -> ArrayT:
        """Sum array elements along `axis`.

        Args:
            a: Input array.
            axis: Reduction axis index.

        Returns:
            Summed array.

        """
        ...


def gather_jax_sequence_logps[ArrayT: JaxSliceableTensorProtocol](
    jnn_mod: JaxNNLogSoftmaxProtocol[ArrayT],
    jnp_mod: JaxNumpyLogpOpsProtocol[ArrayT],
    logits: ArrayT,
    labels: ArrayT,
    *,
    pad_token_id: int = 0,
    ignore_index: int = -100,
) -> ArrayT:
    """Gather causal shifted next-token log-probabilities (`logits[:, :-1]` vs `labels[:, 1:]`) with pad/prompt masking.

    When `seq_len > 1` and `logits.shape[1] == labels.shape[1]`, aligns `logits[:, :-1, :]`
    with `labels[:, 1:]` so position `t` scores token `labels[:, t + 1]`. Tokens matching
    `pad_token_id` or `ignore_index` are masked out before summing per-sequence log-probabilities.

    Args:
        jnn_mod: Loaded `jax.nn` module providing `log_softmax`.
        jnp_mod: Loaded `jax.numpy` module providing gather and reduction ops.
        logits: Vocabulary logits array of shape `(batch_size, seq_len, vocab_size)`.
        labels: Target token ID array of shape `(batch_size, seq_len)`.
        pad_token_id: Padding token ID excluded from summed log-probabilities (default `0`).
        ignore_index: Masked prompt/ignore label value excluded from summed log-probabilities (default `-100`).

    Returns:
        Summed causal sequence log-probabilities of shape `(batch_size,)`.

    """
    if len(logits.shape) >= 3 and len(labels.shape) >= 2 and logits.shape[1] == labels.shape[1] and logits.shape[1] > 1:
        shifted_logits: ArrayT = logits[:, :-1, :]
        shifted_labels: ArrayT = labels[:, 1:]
    else:
        shifted_logits = logits
        shifted_labels = labels

    not_pad = jnp_mod.not_equal(shifted_labels, pad_token_id)
    not_ignore = jnp_mod.not_equal(shifted_labels, ignore_index)
    valid_mask: ArrayT = not_pad * not_ignore
    safe_labels: ArrayT = shifted_labels * valid_mask

    log_probs = jnn_mod.log_softmax(shifted_logits, axis=-1)
    labels_expanded = jnp_mod.expand_dims(safe_labels, axis=-1)
    selected_log_probs = jnp_mod.squeeze(jnp_mod.take_along_axis(log_probs, labels_expanded, axis=-1), axis=-1)
    weighted: ArrayT = selected_log_probs * valid_mask
    return jnp_mod.sum(weighted, axis=-1)


@runtime_checkable
class BeamSequenceArrayProtocol(SupportsFloat, Protocol):
    """Protocol for 2D token ID sequence arrays indexed by `(row, col)` or an index array."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide the array dimensions.

        Returns:
            Tuple of dimension lengths.

        """
        ...

    def __getitem__(
        self,
        item: int | slice | types.EllipsisType | Self | tuple[int | slice | types.EllipsisType, ...],
        /,
    ) -> Self:
        """Index into the sequence array.

        Args:
            item: Index, slice, or tuple of slices.

        Returns:
            Indexed subarray.

        """
        ...

    def reshape(self, *shape: int) -> Self:
        """Reshape the array.

        Args:
            *shape: Target dimension lengths.

        Returns:
            Reshaped array.

        """
        ...


@runtime_checkable
class JaxNNBeamNamespaceProtocol[ArrayT](Protocol):
    """Protocol for `jax.nn` providing `log_softmax`."""

    def log_softmax(self, x: ArrayT, axis: int = -1) -> ArrayT:
        """Compute log-softmax along `axis`.

        Args:
            x: Input logits array.
            axis: Normalization axis index.

        Returns:
            Log-probabilities array.

        """
        ...


@runtime_checkable
class JaxModuleBeamProtocol[ArrayT](Protocol):
    """Protocol for the `jax` module exposing `.nn.log_softmax`."""

    @property
    def nn(self) -> JaxNNBeamNamespaceProtocol[ArrayT]:
        """Provide the `jax.nn` submodule.

        Returns:
            `jax.nn` namespace.

        """
        ...


@runtime_checkable
class JaxNumpyBeamProtocol[ArrayT](Protocol):
    """Protocol for `jax.numpy` providing `argsort` and `concatenate`."""

    def argsort(self, a: ArrayT, axis: int = -1) -> ArrayT:
        """Return the indices that would sort an array.

        Args:
            a: Input 1D array.
            axis: Axis along which to sort.

        Returns:
            Sorted index array.

        """
        ...

    def concatenate(self, arrays: Sequence[ArrayT], axis: int = -1) -> ArrayT:
        """Join a sequence of arrays along an existing axis.

        Args:
            arrays: Sequence of arrays to concatenate.
            axis: Concatenation axis index.

        Returns:
            Concatenated array.

        """
        ...


@runtime_checkable
class JaxJitBeamModuleProtocol(Protocol):
    """Protocol for a `jax` module exposing `.jit` for beam step probability compilation."""

    def jit[ArrayT: BeamSequenceArrayProtocol](
        self,
        fun: Callable[[ArrayT, int], tuple[ArrayT, ArrayT]],
        *,
        static_argnums: tuple[int, ...],
    ) -> Callable[[ArrayT, int], tuple[ArrayT, ArrayT]]:
        """JIT-compile a step probability function.

        Args:
            fun: Pure step-probability function to compile.
            static_argnums: Positional indices treated as static compile-time constants.

        Returns:
            JIT-compiled step-probability callable.

        """
        ...


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
    top_probs: ArrayT = log_probs[top_indices]
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
        prob_val = extract_scalar_float(raw_prob)
        new_score = score + prob_val
        new_beams.append((new_seq, new_score))
    return new_beams


def build_jax_beam_step_fn[ArrayT: BeamSequenceArrayProtocol](
    jax_mod: JaxModuleBeamProtocol[ArrayT],
    jnp_mod: JaxNumpyBeamProtocol[ArrayT],
) -> Callable[[ArrayT, float, ArrayT, int], list[tuple[ArrayT, float]]]:
    """Compile a reusable beam-search step expander once outside the autoregressive decoding loop.

    Hoists `jax.jit` compilation outside per-token decoding steps so XLA traces once per `(shape, beam_width)`
    rather than re-tracing a fresh closure on every generated token.

    Args:
        jax_mod: Loaded `jax` module.
        jnp_mod: Loaded `jax.numpy` module.

    Returns:
        Reusable `(seq, score, logits, beam_width) -> list[(expanded_seq, new_score)]` callable.

    """

    def _step_probs(l_arr: ArrayT, bw: int) -> tuple[ArrayT, ArrayT]:
        """Compute top-k indices and log-probabilities for XLA JIT compilation.

        Args:
            l_arr: Vocabulary logits array for the current decoding step.
            bw: Beam width integer.

        Returns:
            Tuple of `(top_indices, top_log_probs)` arrays.

        """
        return compute_jax_step_probs(l_arr, bw, jax_mod, jnp_mod)

    compiled_probs: Callable[[ArrayT, int], tuple[ArrayT, ArrayT]] = (
        jax_mod.jit(_step_probs, static_argnums=(1,)) if isinstance(jax_mod, JaxJitBeamModuleProtocol) else _step_probs
    )

    def _step(
        seq: ArrayT,
        score: float,
        logits: ArrayT,
        beam_width: int,
    ) -> list[tuple[ArrayT, float]]:
        """Evaluate compiled top-k probabilities and expand `seq` into `beam_width` beams.

        Args:
            seq: Current token ID sequence array.
            score: Cumulative log-probability score of `seq`.
            logits: Vocabulary logits array evaluated for `seq`.
            beam_width: Number of beams to expand.

        Returns:
            List of `(expanded_sequence, new_score)` tuples.

        """
        top_indices, top_probs = compiled_probs(logits, beam_width)
        return expand_jax_beams(seq, score, top_indices, top_probs, beam_width, jnp_mod)

    return _step


def run_jax_beam_search_step[ArrayT: BeamSequenceArrayProtocol](
    seq: ArrayT,
    score: float,
    logits: ArrayT,
    beam_width: int,
    jax_mod: JaxModuleBeamProtocol[ArrayT],
    jnp_mod: JaxNumpyBeamProtocol[ArrayT],
) -> list[tuple[ArrayT, float]]:
    """Compute top-k log probabilities and expand `seq` into `beam_width` beams.

    Args:
        seq: Current token ID sequence array of shape `(1, seq_len)`.
        score: Cumulative log-probability score of `seq`.
        logits: Vocabulary logits array evaluated for `seq`.
        beam_width: Number of beams to expand.
        jax_mod: Loaded `jax` module.
        jnp_mod: Loaded `jax.numpy` module.

    Returns:
        List of `(expanded_sequence, new_score)` tuples.

    """
    step_fn = build_jax_beam_step_fn(jax_mod, jnp_mod)
    return step_fn(seq, score, logits, beam_width)


_SIGNATURE_TRAITS_CACHE: weakref.WeakKeyDictionary[Hashable, tuple[bool, bool, frozenset[str]]] = (
    weakref.WeakKeyDictionary()
)


def _compute_signature_traits[OutT](target: Callable[..., OutT]) -> tuple[bool, bool, frozenset[str]]:
    """Inspect `inspect.signature(target)` to extract positional and keyword acceptance traits.

    Args:
        target: Callable model or unbound function.

    Returns:
        Tuple of `(accepts_positions, has_var_kw, accepted_param_names)`.

    """
    sig = inspect.signature(target)
    params = list(sig.parameters.values())
    has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    pos_params = [
        p for p in params if p.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    accepts_positions = has_var_pos or len(pos_params) >= 2 or "positions" in sig.parameters
    return (accepts_positions, has_var_kw, frozenset(sig.parameters.keys()))


def _get_callable_signature_traits[OutT](model: Callable[..., OutT]) -> tuple[bool, bool, frozenset[str]]:
    """Inspect or retrieve cached `(accepts_positions, has_var_kw, param_names)` via `WeakKeyDictionary`.

    Unwraps bound methods (`types.MethodType`) to cache against the underlying function or instance
    without `id()`-keyed dictionaries or ABA identity-reuse hazards.

    Args:
        model: Callable JAX/Flax NNX model, function, or bound method.

    Returns:
        Tuple of `(accepts_positions, has_var_kw, accepted_param_names)`.

    """
    if isinstance(model, types.MethodType):
        return _compute_signature_traits(model)
    cache_key: Hashable | None = model

    try:
        cached = _SIGNATURE_TRAITS_CACHE.get(cache_key)
        if cached is not None:
            return cached
    except TypeError:
        cache_key = None

    traits = _compute_signature_traits(model)
    if cache_key is not None:
        _SIGNATURE_TRAITS_CACHE[cache_key] = traits
    return traits


def call_jax_causal_lm[OutT](
    model: Callable[..., OutT],
    inputs: ForwardArg,
    positions: ForwardArg = None,
    **extra_kwargs: ForwardArg,
) -> OutT:
    """Invoke a JAX causal language model with signature-aware positional and keyword argument forwarding.

    Args:
        model: Callable JAX/Flax NNX model or function.
        inputs: Input token ID array.
        positions: Optional token position index array.
        **extra_kwargs: Optional multimodal or forward keyword arguments (`pixel_values`, `audio_values`).

    Returns:
        Model output logits array or `(logits, cache)` tuple.

    """
    accepts_positions, has_var_kw, param_names = _get_callable_signature_traits(model)
    accepted_kwargs = extra_kwargs if has_var_kw else {k: v for k, v in extra_kwargs.items() if k in param_names}

    if positions is not None and accepts_positions:
        return model(inputs, positions, **accepted_kwargs)
    return model(inputs, **accepted_kwargs)


@runtime_checkable
class NNXOptimizerProtocol[ModelT, GradsT, UpdateOutT](Protocol):
    """Protocol for a Flax NNX Optimizer supporting `.update(model, grads)`."""

    def update(self, model: ModelT, grads: GradsT, /) -> UpdateOutT:
        """Apply gradients to model parameters.

        Args:
            model: Stateful Flax NNX module being updated.
            grads: Computed parameter gradient PyTree.

        Returns:
            Optional parameter updates PyTree or `None`.

        """
        ...


@runtime_checkable
class NNXModuleNamespaceProtocol[ModelT, TxT, OptT](Protocol):
    """Protocol for the `flax.nnx` module providing `Optimizer` and `Param`."""

    @property
    def Param(self) -> type | str:
        """Provide the `nnx.Param` variable filter type.

        Returns:
            `nnx.Param` filter class or identifier.

        """
        ...

    def Optimizer(self, model: ModelT, tx: TxT, *, wrt: type | str) -> OptT:
        """Construct an `nnx.Optimizer`.

        Args:
            model: Stateful Flax NNX model to optimize.
            tx: Optax gradient transformation.
            wrt: Variable filter (`nnx.Param`).

        Returns:
            Initialized `nnx.Optimizer` instance.

        """
        ...


def create_nnx_optimizer[ModelT, TxT, OptT](
    nnx_module: NNXModuleNamespaceProtocol[ModelT, TxT, OptT],
    model: ModelT,
    tx: TxT,
) -> OptT:
    """Create a Flax NNX Optimizer (`nnx.Optimizer(model, tx, wrt=nnx.Param)`).

    Args:
        nnx_module: Loaded `flax.nnx` module.
        model: Flax NNX module to optimize.
        tx: Optax gradient transformation.

    Returns:
        Configured `nnx.Optimizer` instance.

    """
    return nnx_module.Optimizer(model, tx, wrt=nnx_module.Param)


def apply_nnx_optimizer_update[ModelT, GradsT, UpdateOutT](
    optimizer: NNXOptimizerProtocol[ModelT, GradsT, UpdateOutT],
    model: ModelT,
    grads: GradsT,
) -> None:
    """Apply gradient updates on an `nnx.Optimizer` (`optimizer.update(model, grads)`).

    Args:
        optimizer: Flax NNX optimizer instance.
        model: Flax NNX module being optimized.
        grads: Computed gradient PyTree.

    """
    _ = optimizer.update(model, grads)


@dataclass(frozen=True)
class JaxTopologyInfo:
    """Multi-host and multi-chip JAX runtime topology metadata.

    Attributes:
        is_distributed: Whether multi-host `jax.distributed` is active.
        process_index: Zero-based worker index (`0..process_count-1`).
        process_count: Total number of participating worker hosts.
        local_device_count: Accelerator chip count visible on the local worker.
        global_device_count: Total accelerator chip count across all workers.
        device_kind: Active primary accelerator kind (`'tpu'`, `'gpu'`, `'cpu'`).

    """

    is_distributed: bool
    process_index: int
    process_count: int
    local_device_count: int
    global_device_count: int
    device_kind: str


@runtime_checkable
class _JaxDistributedInitNamespaceProtocol(Protocol):
    """Protocol for `jax.distributed` exposing `.initialize(**kwargs)`."""

    def initialize(self, **kwargs: str | int | None) -> None:
        """Initialize JAX distributed coordination."""
        ...


@runtime_checkable
class _JaxWithDistributedProtocol(Protocol):
    """Protocol for `jax` exposing `.distributed`."""

    @property
    def distributed(self) -> _JaxDistributedInitNamespaceProtocol:
        """Provide `jax.distributed` namespace."""
        ...


@runtime_checkable
class _JaxProcessTopologyProtocol(Protocol):
    """Protocol for `jax` exposing multi-host process/device count queries."""

    def process_index(self) -> int:
        """Return zero-based host rank."""
        ...

    def process_count(self) -> int:
        """Return total host process count."""
        ...

    def local_device_count(self) -> int:
        """Return local device count."""
        ...

    def device_count(self) -> int:
        """Return global device count."""
        ...


@runtime_checkable
class JaxMemoryStatsDeviceProtocol(Protocol):
    """Protocol for a JAX device exposing `.memory_stats()`."""

    def memory_stats(self) -> Mapping[str, int | float] | None:
        """Return device HBM/VRAM usage dictionary or `None`."""
        ...


@runtime_checkable
class _JaxShardingNamespaceProtocol(Protocol):
    """Protocol for `jax.sharding` providing `Mesh`, `NamedSharding`, and `PartitionSpec`."""

    def Mesh(
        self,
        devices: Sequence[JaxMemoryStatsDeviceProtocol | str],
        axis_names: tuple[str, ...],
        /,
    ) -> str | JaxMemoryStatsDeviceProtocol:
        """Construct a device `Mesh`."""
        ...

    def PartitionSpec(self, *axes: str | None) -> str | tuple[str | None, ...]:
        """Construct a `PartitionSpec`."""
        ...

    def NamedSharding(
        self,
        mesh: str | JaxMemoryStatsDeviceProtocol,
        spec: str | tuple[str | None, ...],
        /,
    ) -> JaxShardingProtocol:
        """Construct a `NamedSharding`."""
        ...


@runtime_checkable
class _JaxShardingDevicesModuleProtocol(Protocol):
    """Protocol for `jax` exposing `.devices(...)` and `.sharding`."""

    def devices(self, backend: str | None = None, /) -> Sequence[JaxMemoryStatsDeviceProtocol | str]:
        """Return JAX devices filtered by `backend`."""
        ...


@runtime_checkable
class _JaxWithShardingAttrProtocol(Protocol):
    """Protocol for `jax` exposing `.sharding`."""

    @property
    def sharding(self) -> _JaxShardingNamespaceProtocol:
        """Provide `jax.sharding` namespace."""
        ...


JaxShardingDevicesModuleProtocol = _JaxShardingDevicesModuleProtocol


def reset_jax_distributed_state() -> None:
    """Reset the module-level JAX distributed initialization state (for unit tests)."""
    _jax_dist_init_state[0] = False


def initialize_jax_distributed_if_configured(
    jax_mod: (
        _JaxWithDistributedProtocol
        | _JaxProcessTopologyProtocol
        | _JaxShardingDevicesModuleProtocol
        | ModuleNamespaceProtocol
        | None
    ),
    *,
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
    force_attempt: bool = False,
) -> JaxTopologyInfo:
    """Idempotently initialize multi-host `jax.distributed` when coordinator env/args are present.

    Inspects `JAX_COORDINATOR_ADDRESS`, `MEGASCALE_COORDINATOR_ADDRESS`, `NUM_PROCESSES`,
    `TPU_NUM_WORKERS`, `PROCESS_ID`, and `TPU_WORKER_ID` when explicit arguments are omitted.

    Args:
        jax_mod: Loaded `jax` module or typed test double.
        coordinator_address: Optional explicit `<ip>:<port>` coordinator endpoint.
        num_processes: Optional explicit total host process count.
        process_id: Optional explicit zero-based host rank.
        force_attempt: If `True`, attempt `jax.distributed.initialize` unconditionally.

    Returns:
        `JaxTopologyInfo` describing active host and device topology.

    """
    coord = (
        coordinator_address
        or os.environ.get("JAX_COORDINATOR_ADDRESS")
        or os.environ.get("MEGASCALE_COORDINATOR_ADDRESS")
        or None
    )
    raw_np = (
        num_processes
        if num_processes is not None
        else (os.environ.get("NUM_PROCESSES") or os.environ.get("TPU_NUM_WORKERS"))
    )
    raw_pid = (
        process_id if process_id is not None else (os.environ.get("PROCESS_ID") or os.environ.get("TPU_WORKER_ID"))
    )
    parsed_np = int(raw_np) if isinstance(raw_np, (int, str)) and str(raw_np).strip() else None
    parsed_pid = int(raw_pid) if isinstance(raw_pid, (int, str)) and str(raw_pid).strip() else None

    has_env_config = coord is not None or parsed_np is not None or parsed_pid is not None
    should_call = force_attempt or (has_env_config and not _jax_dist_init_state[0])

    init_succeeded = _jax_dist_init_state[0]
    if should_call and isinstance(jax_mod, _JaxWithDistributedProtocol):
        try:
            init_kwargs: dict[str, str | int] = {}
            if coord is not None:
                init_kwargs["coordinator_address"] = coord
            if parsed_np is not None:
                init_kwargs["num_processes"] = parsed_np
            if parsed_pid is not None:
                init_kwargs["process_id"] = parsed_pid
            jax_mod.distributed.initialize(**init_kwargs)
            if not force_attempt:
                _jax_dist_init_state[0] = True
            init_succeeded = True
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as init_err:
            logger.warning("jax.distributed.initialize() failed or already initialized: %s", init_err)

    proc_idx = 0
    proc_cnt = 1
    local_cnt = 1
    global_cnt = 1
    if isinstance(jax_mod, _JaxProcessTopologyProtocol):
        try:
            proc_idx = int(jax_mod.process_index())
            proc_cnt = max(1, int(jax_mod.process_count()))
            local_cnt = max(1, int(jax_mod.local_device_count()))
            global_cnt = max(1, int(jax_mod.device_count()))
        except RuntimeError:
            pass

    dev_kind = "cpu"
    if isinstance(jax_mod, _JaxShardingDevicesModuleProtocol):
        for candidate in ("tpu", "gpu"):
            try:
                if len(jax_mod.devices(candidate)) > 0:
                    dev_kind = candidate
                    break
            except RuntimeError:
                continue

    return JaxTopologyInfo(
        is_distributed=bool(init_succeeded or proc_cnt > 1),
        process_index=proc_idx,
        process_count=proc_cnt,
        local_device_count=local_cnt,
        global_device_count=global_cnt,
        device_kind=dev_kind,
    )


def build_tpu_mesh_and_sharding[DevT](
    jax_mod: _JaxShardingDevicesModuleProtocol | None,
    hardware: str,
    *,
    batch_size: int = 1,
    tensor_parallel_size: int = 1,
    strict_hardware: bool = False,
    backend_label: str = "jax",
    fallback_device: DevT | None = None,
) -> tuple[
    JaxMemoryStatsDeviceProtocol | str | DevT | None,
    list[JaxMemoryStatsDeviceProtocol | str],
    str,
    JaxShardingProtocol | None,
]:
    """Resolve TPU/GPU/CPU devices and build a multi-chip `NamedSharding` across all chips when available.

    Args:
        jax_mod: Loaded `jax` module or typed test double.
        hardware: Target hardware identifier (`'tpu'`, `'tpu-v6e'`, `'tpu-v5p'`, `'gpu'`, `'cpu'`).
        batch_size: Batch size used to determine data-parallel vs replicated input sharding.
        tensor_parallel_size: Requested tensor-parallel degree (`>= 1`).
        strict_hardware: Whether to raise `HardwareUnavailableError` if requested TPU/GPU is absent.
        backend_label: Backend identifier recorded on `HardwareUnavailableError`.
        fallback_device: Optional fallback device when `jax_mod` is `None`.

    Returns:
        4-tuple of `(primary_device, active_devices, resolved_device_str, data_sharding_or_none)`.

    Raises:
        HardwareUnavailableError: When `strict_hardware=True` and requested TPU/GPU is unavailable.

    """
    _ = tensor_parallel_size
    if jax_mod is None:
        if strict_hardware and normalize_hardware_family(hardware) in {"tpu", "gpu"}:
            msg = f"JAX module unavailable for requested hardware '{hardware}'."
            raise HardwareUnavailableError(msg, hardware=hardware, backend=backend_label)
        return (fallback_device, [], "cpu", None)

    hw_family = normalize_hardware_family(hardware)
    active_devs: list[JaxMemoryStatsDeviceProtocol | str] = []
    resolved_str = "cpu"

    if hw_family in {"tpu", "gpu"}:
        try:
            found = list(jax_mod.devices(hw_family))
        except RuntimeError:
            found = []
        if found:
            active_devs = found
            resolved_str = f"{hw_family}:0"
        elif strict_hardware:
            msg = f"JAX backend '{backend_label}' could not resolve '{hardware}' devices."
            raise HardwareUnavailableError(msg, hardware=hardware, backend=backend_label)

    if not active_devs:
        try:
            active_devs = list(jax_mod.devices("cpu"))
        except RuntimeError:
            active_devs = []
        resolved_str = "cpu"

    primary: JaxMemoryStatsDeviceProtocol | str | DevT | None = active_devs[0] if active_devs else fallback_device
    sharding: JaxShardingProtocol | None = None
    if len(active_devs) > 1 and isinstance(jax_mod, _JaxWithShardingAttrProtocol):
        try:
            mesh = jax_mod.sharding.Mesh(active_devs, ("data",))
            spec = (
                jax_mod.sharding.PartitionSpec("data", None)
                if batch_size >= len(active_devs) and batch_size % len(active_devs) == 0
                else jax_mod.sharding.PartitionSpec()
            )
            sharding = jax_mod.sharding.NamedSharding(mesh, spec)
        except (TypeError, ValueError, RuntimeError, AttributeError):
            sharding = None

    return (primary, active_devs, resolved_str, sharding)


def aggregate_jax_peak_hbm_mb(
    devices: Sequence[JaxMemoryStatsDeviceProtocol | str],
) -> float | None:
    """Return maximum per-device `peak_bytes_in_use` in MiB across `devices`, or `None` if unavailable.

    Args:
        devices: Sequence of JAX device handles or strings.

    Returns:
        Peak HBM usage in MiB when device memory stats are reported, else `None`.

    """
    peaks: list[float] = []
    for dev in devices:
        if isinstance(dev, JaxMemoryStatsDeviceProtocol):
            try:
                stats = dev.memory_stats()
            except (AttributeError, KeyError, RuntimeError, TypeError):
                stats = None
            if stats is not None:
                raw = stats.get("peak_bytes_in_use", stats.get("bytes_in_use"))
                if raw is not None and raw > 0:
                    peaks.append(float(raw) / (1024.0 * 1024.0))
    return max(peaks) if peaks else None


__all__ = [
    "BeamSequenceArrayProtocol",
    "JaxJitBeamModuleProtocol",
    "JaxMemoryStatsDeviceProtocol",
    "JaxModuleBeamProtocol",
    "JaxNNBeamNamespaceProtocol",
    "JaxNNLogSoftmaxProtocol",
    "JaxNumpyBeamProtocol",
    "JaxNumpyLogpOpsProtocol",
    "JaxShardingDevicesModuleProtocol",
    "JaxSliceableTensorProtocol",
    "JaxTopologyInfo",
    "NNXModuleNamespaceProtocol",
    "NNXOptimizerProtocol",
    "ScalarItemExtractorProtocol",
    "aggregate_jax_peak_hbm_mb",
    "apply_nnx_optimizer_update",
    "build_jax_beam_step_fn",
    "build_tpu_mesh_and_sharding",
    "call_jax_causal_lm",
    "compute_jax_step_probs",
    "create_nnx_optimizer",
    "expand_jax_beams",
    "gather_jax_sequence_logps",
    "initialize_jax_distributed_if_configured",
    "reset_jax_distributed_state",
    "run_jax_beam_search_step",
]
