"""JAX-specific benchmarking pipeline."""

from __future__ import annotations

import contextlib
import contextvars
import logging
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import jax
from flax import nnx
from jax import Array

from gemma_4_sql.backends._jax_common import (
    JaxMemoryStatsDeviceProtocol,
    aggregate_jax_peak_hbm_mb,
    build_tpu_mesh_and_sharding,
    initialize_jax_distributed_if_configured,
)
from gemma_4_sql.backends.common_benchmark import (
    BenchmarkTelemetry,
    DetailedLoopMetrics,
    execute_detailed_benchmark_loop,
    get_current_rss_mb,
    normalize_benchmark_mode,
    normalize_hardware_family,
    run_benchmark_wrapper,
)
from gemma_4_sql.backends.jax._protocols import is_jnp_missing
from gemma_4_sql.backends.jax.gemma4._jnp import jnp
from gemma_4_sql.backends.jax.gemma4.config import JaxDTypeLike, ModelConfig, resolve_gemma4_config
from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import BenchmarkResultDict, JaxArrayProtocol, JaxCausalLMProtocol, JSONValue

logger = logging.getLogger(__name__)

Gemma4Config = ModelConfig


type _JaxBenchmarkModel = JaxCausalLMProtocol | Gemma4ForCausalLM | Callable[..., Array]


JaxDeviceProtocol = JaxMemoryStatsDeviceProtocol


@runtime_checkable
class _JaxDevicesModuleProtocol(Protocol):
    """Protocol for a `jax` module exposing `.devices(...)`."""

    def devices(self, backend: str | None = None, /) -> Sequence[JaxDeviceProtocol]:
        """Return available JAX devices for `backend`.

        Args:
            backend: Optional accelerator platform (`'tpu'`, `'gpu'`, `'cpu'`).

        Returns:
            Sequence of matching JAX devices.

        """
        ...


@runtime_checkable
class _NNXJitProtocol(Protocol):
    """Protocol for a `flax.nnx` module exposing `.jit`."""

    def jit(
        self,
        fn: Callable[[_JaxBenchmarkModel, Array], Array],
        /,
    ) -> Callable[[_JaxBenchmarkModel, Array], Array]:
        """JIT-compile a benchmark pass function."""
        ...


@runtime_checkable
class _JaxBlockUntilReadyProtocol(Protocol):
    """Protocol for a `jax` module exposing `.block_until_ready`."""

    def block_until_ready(self, x: Array, /) -> Array:
        """Block until asynchronous JAX array computation completes."""
        ...


def _resolve_jax_dtype(dtype_str: str) -> JaxDTypeLike:
    """Resolve a string dtype identifier to a `jax.numpy` dtype.

    Args:
        dtype_str: Dtype name such as `'bfloat16'`, `'float16'`, or `'float32'`.

    Returns:
        Resolved `JaxDTypeLike` object.

    """
    dtype_map: dict[str, JaxDTypeLike] = {
        "bfloat16": jnp.bfloat16 or "bfloat16",
        "float16": jnp.float16,
        "float32": jnp.float32,
        "int32": jnp.int32,
        "int8": jnp.int8,
    }
    return dtype_map.get(dtype_str, jnp.bfloat16 or "bfloat16")


@runtime_checkable
class _JaxDevicePutProtocol(Protocol):
    """Protocol for `jax` exposing `device_put(x, device_or_sharding)`."""

    def device_put[ArrT](self, x: ArrT, device: JaxShardingProtocol, /) -> ArrT:
        """Place or shard `x` according to `device` (`NamedSharding`)."""
        ...


from gemma_4_sql.type_hints import JaxShardingProtocol  # ruff: ignore[module-import-not-at-top-of-file]

_LAST_JAX_METRICS: contextvars.ContextVar[DetailedLoopMetrics | None] = contextvars.ContextVar(
    "jax_last_benchmark_metrics",
    default=None,
)


def _get_device(hardware: str) -> JaxDeviceProtocol | str | None:
    """Provide the JAX device for the requested hardware type (`'tpu'`, `'tpu-v6e'`, `'gpu'`, `'cpu'`).

    Args:
        hardware: Target hardware type (`'tpu'`, `'tpu-v5p'`, `'gpu'`, `'cpu'`).

    Returns:
        The matched JAX device or default CPU device.

    """
    if not isinstance(jax, _JaxDevicesModuleProtocol):
        return None
    hw_family = normalize_hardware_family(hardware)
    try:
        if hw_family == "tpu" and jax.devices("tpu"):
            return jax.devices("tpu")[0]
        if hw_family == "gpu" and jax.devices("gpu"):
            return jax.devices("gpu")[0]
    except RuntimeError:
        pass
    return jax.devices("cpu")[0]


def _run_benchmark_pass(
    model: _JaxBenchmarkModel,
    batch_size: int,
    num_runs: int,
    warmup_steps: int,
    mode: str,
    max_new_tokens: int,
    device: JaxDeviceProtocol | str | None,
    *,
    sharding: JaxShardingProtocol | None = None,
) -> tuple[float, float, float]:
    """Execute the forward or generation benchmark loop on `device` or multi-chip `sharding`.

    Args:
        model: The `JaxCausalLMProtocol` model to benchmark.
        batch_size: Number of items in a single batch.
        num_runs: Number of timed iterations.
        warmup_steps: Number of untimed warmup iterations.
        mode: Benchmark mode (`'prefill'`, `'decode'`, `'generate'`, or `'end-to-end'`).
        max_new_tokens: Number of tokens to generate in non-prefill modes.
        device: Target JAX device.
        sharding: Optional multi-chip `NamedSharding` across TPU/GPU `Mesh`.

    Returns:
        A 3-tuple of `(tokens_per_sec, latency_ms, memory_mb)`.

    """
    canon_mode = normalize_benchmark_mode(mode, backend="jax")

    def forward_pass(m: _JaxBenchmarkModel, inputs: Array) -> Array:
        """Execute a single forward pass.

        Args:
            m: Model to run.
            inputs: Dummy token ID inputs.

        Returns:
            Model forward pass output array.

        """
        out = m(inputs)
        return jnp.asarray(out) if not isinstance(out, Array) else out

    def generate_pass(m: _JaxBenchmarkModel, inputs: Array) -> Array:
        """Execute an unrolled autoregressive generation pass.

        Args:
            m: Model to run.
            inputs: Initial token sequence array.

        Returns:
            Generated token sequence array.

        """
        seq = inputs
        for _ in range(max_new_tokens):
            positions = jnp.arange(seq.shape[1])[None, :]
            raw_logits = m(seq, positions)
            logits = jnp.asarray(raw_logits) if not isinstance(raw_logits, Array) else raw_logits
            token = jnp.argmax(logits[..., -1, :], axis=-1, keepdims=True)
            seq = jnp.concatenate([seq, token], axis=-1)
        return seq

    compiled_forward = nnx.jit(forward_pass) if isinstance(nnx, _NNXJitProtocol) else forward_pass
    compiled_generate = nnx.jit(generate_pass) if isinstance(nnx, _NNXJitProtocol) else generate_pass

    default_dev_fn: Callable[[JaxDeviceProtocol | str | None], contextlib.AbstractContextManager[None]] = (
        jax.default_device
    )
    dev_cm = contextlib.nullcontext() if sharding is not None else default_dev_fn(device)
    with dev_cm:
        dummy_inputs = jax.random.randint(jax.random.key(42), (batch_size, 32), 1, 256000, dtype=jnp.int32)
        if sharding is not None and isinstance(jax, _JaxDevicePutProtocol):
            dummy_inputs = jax.device_put(dummy_inputs, sharding)

        def _step() -> None:
            """Execute one compiled JAX step and synchronize device output."""
            out = (
                compiled_forward(model, dummy_inputs)
                if canon_mode == "prefill"
                else compiled_generate(model, dummy_inputs)
            )
            if isinstance(jax, _JaxBlockUntilReadyProtocol):
                jax.block_until_ready(out)

        metrics = execute_detailed_benchmark_loop(
            _step,
            warmup_steps=warmup_steps,
            num_runs=num_runs,
            tokens_per_iter=32 if canon_mode == "prefill" else max_new_tokens,
            batch_size=batch_size,
        )
        _LAST_JAX_METRICS.set(metrics)

        memory_mb = get_current_rss_mb()
        try:
            stats = (device.memory_stats() if isinstance(device, JaxDeviceProtocol) else None) or {}
            memory_mb = float(stats.get("peak_bytes_in_use", memory_mb * 1024.0 * 1024.0)) / (1024.0 * 1024.0)
        except (AttributeError, KeyError, RuntimeError, TypeError):
            memory_mb = get_current_rss_mb()

    return (float(metrics.tokens_per_sec), float(metrics.latency_ms), float(memory_mb))


def benchmark_model(
    model_name: str,
    hardware: str = "tpu-v5p",
    batch_size: int = 32,
    **kwargs: JSONValue,
) -> BenchmarkResultDict:
    """Benchmark a model using the native JAX backend (`jax` or `jax_native`).

    Args:
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., `'tpu'`, `'tpu-v6e'`, `'gpu'`, `'cpu'`).
        batch_size: Batch size to use during benchmarking.
        **kwargs: Additional args like `num_runs`, `dtype`, `mode`, `max_new_tokens`, `warmup_steps`,
            `backend_alias`, `strict_hardware`, `tensor_parallel_size`.

    Returns:
        A dictionary containing benchmark metrics and status.

    Raises:
        DependencyMissingError: If required JAX dependencies are missing.

    """
    backend_alias = str(kwargs.get("backend_alias", "jax"))
    strict_hw = bool(kwargs.get("strict_hardware", False))
    tp_size = int(str(kwargs.get("tensor_parallel_size", 1)))

    if jax is None or is_jnp_missing(jnp) or nnx is None or not callable(Gemma4ForCausalLM):
        raise DependencyMissingError("JAX dependencies are missing.")

    def _run() -> BenchmarkTelemetry:
        """Initialize distributed JAX if configured, shard across TPU Mesh, and run benchmark.

        Returns:
            `BenchmarkTelemetry` across timed iterations.

        """
        _ = initialize_jax_distributed_if_configured(jax)
        dtype_str = str(kwargs.get("dtype", "bfloat16"))
        mode = str(kwargs.get("mode", "prefill"))
        max_new_tokens = int(str(kwargs.get("max_new_tokens", 128)))
        warmup_steps = int(str(kwargs.get("warmup_steps", 5)))
        num_runs = int(str(kwargs.get("num_runs", 5)))

        target_dtype = _resolve_jax_dtype(dtype_str)

        config = resolve_gemma4_config(Gemma4Config, model_name)
        config.dtype = target_dtype
        model = Gemma4ForCausalLM(config, rngs=nnx.Rngs(0))

        primary_dev, active_devs, resolved_str, sharding = build_tpu_mesh_and_sharding(
            jax if isinstance(jax, _JaxDevicesModuleProtocol) else None,
            hardware,
            batch_size=batch_size,
            tensor_parallel_size=tp_size,
            strict_hardware=strict_hw,
            backend_label=backend_alias,
        )
        bench_dev: JaxDeviceProtocol | str | None = (
            primary_dev if isinstance(primary_dev, (JaxDeviceProtocol, str)) else _get_device(hardware)
        )

        ctx_tok = _LAST_JAX_METRICS.set(None)
        try:
            tps, lat_ms, mem_mb = _run_benchmark_pass(
                model,
                batch_size,
                num_runs,
                warmup_steps,
                mode,
                max_new_tokens,
                bench_dev,
                sharding=sharding,
            )
            loop_metrics = _LAST_JAX_METRICS.get()
        finally:
            _LAST_JAX_METRICS.reset(ctx_tok)

        agg_hbm = aggregate_jax_peak_hbm_mb(active_devs) if resolved_str.startswith(("tpu", "gpu")) else None
        effective_mem = agg_hbm if agg_hbm is not None else mem_mb
        return BenchmarkTelemetry(
            tokens_per_sec=tps,
            latency_ms=lat_ms,
            memory_mb=effective_mem,
            resolved_device=resolved_str,
            device_count=max(1, len(active_devs)),
            compile_latency_ms=loop_metrics.compile_latency_ms if loop_metrics is not None else None,
            peak_hbm_mb=agg_hbm,
            latency_samples=loop_metrics.latency_samples if loop_metrics is not None else (),
        )

    return run_benchmark_wrapper(
        backend_name=backend_alias,
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        benchmark_fn=_run,
        strict_hardware=strict_hw,
        raise_on_error=bool(kwargs["raise_on_error"]) if "raise_on_error" in kwargs else None,
    )


__all__ = [
    "Gemma4Config",
    "Gemma4ForCausalLM",
    "JaxArrayProtocol",
    "JaxDeviceProtocol",
    "benchmark_model",
]
