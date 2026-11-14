"""MaxText-specific benchmarking pipeline."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from types import EllipsisType
from typing import Protocol, runtime_checkable

from gemma_4_sql._types_models import MaxTextModelConstructorProtocol
from gemma_4_sql.backends.common_benchmark import (
    execute_timed_benchmark_loop,
    get_current_rss_mb,
    run_benchmark_wrapper,
)
from gemma_4_sql.backends.lazy_loader import ModuleNamespaceProtocol, import_optional_symbol, optional_import
from gemma_4_sql.backends.maxtext._protocols import JaxDistributedHolderProtocol
from gemma_4_sql.type_hints import (
    BenchmarkResultDict,
    DTypeProtocol,
    ForwardOutput,
    JaxArrayProtocol,
    JSONValue,
    MaxTextCausalLMProtocol,
    PyTreeNode,
    TensorProtocol,
    TensorType,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _MemoryStatsDeviceProtocol(Protocol):
    """Protocol for a JAX device exposing `.memory_stats()`."""

    def memory_stats(self) -> Mapping[str, int] | None:
        """Return memory usage statistics for the device.

        Returns:
            Mapping of memory stat names to byte counts, or `None`.

        """
        ...


@runtime_checkable
class _JaxJitAndSyncProtocol(Protocol):
    """Protocol for a `jax` module providing `.jit(...)` and `.block_until_ready(...)`."""

    def jit[FnT](self, fun: FnT, /) -> FnT:
        """JIT-compile a function.

        Args:
            fun: Callable to compile.

        Returns:
            Compiled callable.

        """
        ...

    def block_until_ready[OutT](self, x: OutT, /) -> OutT:
        """Block until asynchronous device computation completes.

        Args:
            x: Output PyTree or array.

        Returns:
            Synchronized output.

        """
        ...


def _as_jit_sync(
    mod: _JaxBenchmarkModuleProtocol | ModuleNamespaceProtocol,
) -> _JaxJitAndSyncProtocol | None:
    """Return `mod` narrowed to `_JaxJitAndSyncProtocol` when `.jit` and `.block_until_ready` are present.

    Args:
        mod: Loaded `jax` module.

    Returns:
        `_JaxJitAndSyncProtocol` instance or `None`.

    """
    return mod if isinstance(mod, _JaxJitAndSyncProtocol) else None


@runtime_checkable
class _JaxBenchmarkRandomProtocol(Protocol):
    """Protocol for `jax.random` exposing `PRNGKey` and `randint`."""

    def PRNGKey(self, seed: int, /) -> JaxArrayProtocol:
        """Create a JAX PRNG key.

        Args:
            seed: Integer seed.

        Returns:
            PRNG key `JaxArrayProtocol`.

        """
        ...

    def randint(
        self,
        key: JaxArrayProtocol,
        shape: tuple[int, int],
        minval: int,
        maxval: int,
        *,
        dtype: DTypeProtocol,
    ) -> JaxArrayProtocol:
        """Sample uniform random integer token IDs.

        Args:
            key: JAX PRNG key.
            shape: `(batch_size, seq_len)` shape tuple.
            minval: Inclusive minimum token ID.
            maxval: Exclusive maximum token ID.
            dtype: Target integer dtype.

        Returns:
            Sampled token ID `JaxArrayProtocol`.

        """
        ...


@runtime_checkable
class _JaxBenchmarkModuleProtocol(JaxDistributedHolderProtocol, Protocol):
    """Protocol for `jax` device discovery, placement, and random token generation in benchmarking."""

    @property
    def random(self) -> _JaxBenchmarkRandomProtocol:
        """Provide `jax.random` namespace."""
        ...

    def devices(self, backend: str | None = None, /) -> Sequence[_MemoryStatsDeviceProtocol]:
        """List visible JAX devices for `backend`.

        Args:
            backend: Optional accelerator platform (`'tpu'`, `'gpu'`, `'cpu'`).

        Returns:
            Sequence of `_MemoryStatsDeviceProtocol` devices.

        """
        ...

    def default_device(self, device: _MemoryStatsDeviceProtocol | str | None, /) -> AbstractContextManager[None]:
        """Return a context manager scoping default JAX device placement.

        Args:
            device: Target JAX device descriptor.

        Returns:
            Device placement context manager.

        """
        ...


@runtime_checkable
class _JnpBenchmarkModuleProtocol(Protocol):
    """Protocol for `jax.numpy` dtypes, `argmax`, and `concatenate` in MaxText benchmarking."""

    @property
    def bfloat16(self) -> DTypeProtocol:
        """Provide `jnp.bfloat16` dtype."""
        ...

    @property
    def float16(self) -> DTypeProtocol:
        """Provide `jnp.float16` dtype."""
        ...

    @property
    def float32(self) -> DTypeProtocol:
        """Provide `jnp.float32` dtype."""
        ...

    @property
    def int32(self) -> DTypeProtocol:
        """Provide `jnp.int32` dtype."""
        ...

    def argmax(
        self,
        a: JaxArrayProtocol | TensorType,
        *,
        axis: int = -1,
        keepdims: bool = False,
    ) -> JaxArrayProtocol:
        """Return argmax token indices along `axis`.

        Args:
            a: Last-step vocabulary logits slice.
            axis: Vocabulary dimension index.
            keepdims: Whether to keep reduced dimension of size 1.

        Returns:
            Selected token `JaxArrayProtocol`.

        """
        ...

    def concatenate(
        self,
        arrays: Sequence[JaxArrayProtocol | TensorType],
        *,
        axis: int = -1,
    ) -> JaxArrayProtocol:
        """Concatenate `arrays` along `axis`.

        Args:
            arrays: Sequence of token arrays.
            axis: Sequence axis index.

        Returns:
            Concatenated `JaxArrayProtocol`.

        """
        ...


jax: _JaxBenchmarkModuleProtocol | None = optional_import("jax", _JaxBenchmarkModuleProtocol)
jnp: _JnpBenchmarkModuleProtocol | None = optional_import("jax.numpy", _JnpBenchmarkModuleProtocol)
Gemma4Model: MaxTextModelConstructorProtocol | None = import_optional_symbol(
    "maxtext.models.gemma4",
    "Gemma4Model",
    MaxTextModelConstructorProtocol,
)


def _resolve_jnp_dtype(dtype_str: str) -> DTypeProtocol | str | None:
    """Resolve a string dtype identifier on `jax.numpy`.

    Args:
        dtype_str: Dtype name such as `'bfloat16'`, `'float16'`, or `'float32'`.

    Returns:
        Resolved `jax.numpy` dtype or `None`.

    """
    assert jnp is not None
    dtype_map: dict[str, DTypeProtocol | str | None] = {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": jnp.float32,
    }
    return dtype_map.get(dtype_str)


def _get_device(hardware: str) -> _MemoryStatsDeviceProtocol | str | None:
    """Provide the JAX device for the requested hardware.

    Args:
        hardware: Target hardware string.

    Returns:
        Matched JAX device or default CPU device.

    """
    try:
        if hardware == "tpu" and jax is not None and jax.devices("tpu"):
            return jax.devices("tpu")[0]
        if hardware == "gpu" and jax is not None and jax.devices("gpu"):
            return jax.devices("gpu")[0]
    except RuntimeError:
        pass
    return jax.devices("cpu")[0] if jax is not None else None


type _MaxTextBenchmarkParams = (
    Mapping[str, PyTreeNode] | Mapping[str, TensorProtocol | Mapping[str, TensorProtocol]] | dict[str, TensorType]
)


@runtime_checkable
class _MaxTextLogitsProtocol(Protocol):
    """Protocol for a 3D logits array supporting `[..., -1, :]` slicing."""

    def __getitem__(
        self,
        idx: tuple[EllipsisType, int, slice],
        /,
    ) -> JaxArrayProtocol | TensorType:
        """Slice the last sequence position logits.

        Args:
            idx: Ellipsis slice tuple `(..., -1, slice(None))`.

        Returns:
            Last-position vocabulary logits.

        """
        ...


@runtime_checkable
class _MaxTextBenchmarkModelProtocol(Protocol):
    """Protocol for a stateless MaxText model supporting `apply(params, inputs)`."""

    def apply(
        self,
        params: _MaxTextBenchmarkParams,
        inputs: TensorType,
    ) -> _MaxTextLogitsProtocol:
        """Apply model parameters to `inputs`.

        Args:
            params: Parameter PyTree dictionary.
            inputs: Input token ID array.

        Returns:
            Output vocabulary logits.

        """
        ...


def _run_benchmark_pass(
    model: MaxTextCausalLMProtocol | _MaxTextBenchmarkModelProtocol,
    params: _MaxTextBenchmarkParams,
    batch_size: int,
    num_runs: int,
    warmup_steps: int,
    mode: str,
    max_new_tokens: int,
    device: _MemoryStatsDeviceProtocol | str | None,
) -> tuple[float, float, float]:
    """Execute the forward pass benchmark loop.

    Args:
        model: The model.
        params: Model parameters.
        batch_size: The number of items to process in a single batch.
        num_runs: The integer value for num runs.
        warmup_steps: Number of warmup steps.
        mode: Benchmark mode.
        max_new_tokens: Max tokens.
        device: JAX device.

    Returns:
        A tuple containing `(tokens_per_sec, latency_ms, memory_mb)`.

    """
    assert jax is not None
    assert jnp is not None

    def forward_pass(params: _MaxTextBenchmarkParams, inputs: TensorType) -> ForwardOutput | _MaxTextLogitsProtocol:
        """Execute model forward pass.

        Args:
            params: Model parameters.
            inputs: Dummy inputs array.

        Returns:
            Forward pass model output.

        """
        return model.apply(params, inputs)

    def generate_pass(params: _MaxTextBenchmarkParams, inputs: TensorType) -> TensorType:
        """Execute simple generation pass.

        Args:
            params: Model parameters.
            inputs: Input sequence array.

        Returns:
            Generated sequence array.

        """
        assert jnp is not None
        seq = inputs
        for _ in range(max_new_tokens):
            logits = model.apply(params, seq)
            token = jnp.argmax(logits[..., -1, :], axis=-1, keepdims=True)
            seq = jnp.concatenate([seq, token], axis=-1)
        return seq

    jit_sync = _as_jit_sync(jax)
    if jit_sync is not None:
        forward_pass = jit_sync.jit(forward_pass)
        generate_pass = jit_sync.jit(generate_pass)

    with jax.default_device(device):
        dummy_inputs = jax.random.randint(jax.random.PRNGKey(42), (batch_size, 32), 1, 256000, dtype=jnp.int32)

        def _step() -> None:
            """Execute one MaxText forward or generate step and synchronize."""
            out = forward_pass(params, dummy_inputs) if mode == "prefill" else generate_pass(params, dummy_inputs)
            if jit_sync is not None:
                jit_sync.block_until_ready(out)

        tokens_per_sec, latency_ms = execute_timed_benchmark_loop(
            _step,
            warmup_steps=warmup_steps,
            num_runs=num_runs,
            tokens_per_iter=32 if mode == "prefill" else max_new_tokens,
            batch_size=batch_size,
        )

        memory_mb = get_current_rss_mb()
        if isinstance(device, _MemoryStatsDeviceProtocol):
            stats = device.memory_stats()
            if stats is not None:
                memory_mb = float(stats.get("peak_bytes_in_use", int(memory_mb * 1024 * 1024))) / (1024 * 1024)

    return (float(tokens_per_sec), float(latency_ms), float(memory_mb))


def benchmark_model(
    model_name: str,
    hardware: str = "tpu-v5p",
    batch_size: int = 32,
    **kwargs: JSONValue,
) -> BenchmarkResultDict:
    """Benchmark a model using the MaxText backend.

    Args:
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., `'tpu-v5p'`, `'gpu'`).
        batch_size: Batch size to use during benchmarking.
        **kwargs: Additional args like `num_runs`.

    Returns:
        A dictionary containing benchmark metrics and status.

    """

    def _run() -> tuple[float, float, float]:
        """Initialize MaxText model and execute hardware benchmark pass.

        Returns:
            Tuple of `(tokens_per_sec, latency_ms, memory_mb)`.

        """
        assert jax is not None
        assert jnp is not None
        assert Gemma4Model is not None
        dtype_str = str(kwargs.get("dtype", "bfloat16"))
        mode = str(kwargs.get("mode", "prefill"))
        max_new_tokens = int(str(kwargs.get("max_new_tokens", 128)))
        warmup_steps = int(str(kwargs.get("warmup_steps", 5)))
        device = _get_device(hardware)

        try:
            jax.distributed.initialize()
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.warning("jax.distributed.initialize() failed: %s", e)

        try:
            model = Gemma4Model(model_name, dtype=_resolve_jnp_dtype(dtype_str))
        except TypeError:
            model = Gemma4Model(model_name)

        rng = jax.random.PRNGKey(0)
        dummy_inputs = jax.random.randint(rng, (batch_size, 32), 1, 256000, dtype=jnp.int32)
        params = model.init(rng, dummy_inputs)

        num_runs = int(str(kwargs.get("num_runs", 5)))
        return _run_benchmark_pass(model, params, batch_size, num_runs, warmup_steps, mode, max_new_tokens, device)

    return run_benchmark_wrapper(
        backend_name="maxtext",
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        missing_deps=jax is None or jnp is None or Gemma4Model is None,
        benchmark_fn=_run,
    )
