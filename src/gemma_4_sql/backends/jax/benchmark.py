"""JAX-specific benchmarking pipeline."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from gemma_4_sql.backends.common_benchmark import get_current_rss_mb, run_benchmark_wrapper
from gemma_4_sql.backends.jax.gemma4.config import JaxDTypeLike, ModelConfig
from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import BenchmarkResultDict, JaxArrayProtocol, JaxCausalLMProtocol, JSONValue

logger = logging.getLogger(__name__)

Gemma4Config = ModelConfig


type _JaxBenchmarkModel = JaxCausalLMProtocol | Gemma4ForCausalLM | Callable[..., Array]


@runtime_checkable
class JaxDeviceProtocol(Protocol):
    """Protocol for a JAX device optionally exposing `.memory_stats()`."""

    def memory_stats(self) -> Mapping[str, int] | None:
        """Return memory usage statistics for the device."""
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
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": jnp.float32,
        "int32": jnp.int32,
        "int8": jnp.int8,
    }
    return dtype_map.get(dtype_str, jnp.bfloat16)  # pyright: ignore[reportReturnType]


def _get_device(hardware: str) -> JaxDeviceProtocol | str | None:
    """Provide the JAX device for the requested hardware type.

    Args:
        hardware: Target hardware type (`'tpu'`, `'gpu'`, `'cpu'`).

    Returns:
        The matched JAX device or default CPU device.

    """
    try:
        if hardware == "tpu" and jax is not None and jax.devices("tpu"):
            return jax.devices("tpu")[0]
        if hardware == "gpu" and jax is not None and jax.devices("gpu"):
            return jax.devices("gpu")[0]
    except RuntimeError:
        pass
    return jax.devices("cpu")[0] if jax is not None else None


def _run_benchmark_pass(
    model: _JaxBenchmarkModel,
    batch_size: int,
    num_runs: int,
    warmup_steps: int,
    mode: str,
    max_new_tokens: int,
    device: JaxDeviceProtocol | str | None,
) -> tuple[float, float, float]:
    """Execute the forward or generation benchmark loop on `device`.

    Args:
        model: The `JaxCausalLMProtocol` model to benchmark.
        batch_size: Number of items in a single batch.
        num_runs: Number of timed iterations.
        warmup_steps: Number of untimed warmup iterations.
        mode: Benchmark mode (`'prefill'` or `'generate'`).
        max_new_tokens: Number of tokens to generate in `'generate'` mode.
        device: Target JAX device.

    Returns:
        A 3-tuple of `(tokens_per_sec, latency_ms, memory_mb)`.

    """

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

    with jax.default_device(device):  # pyright: ignore[reportArgumentType]
        dummy_inputs = jax.random.randint(jax.random.key(42), (batch_size, 32), 1, 256000, dtype=jnp.int32)

        for _ in range(warmup_steps):
            out = compiled_forward(model, dummy_inputs) if mode == "prefill" else compiled_generate(model, dummy_inputs)
            if isinstance(jax, _JaxBlockUntilReadyProtocol):
                jax.block_until_ready(out)

        start_time = time.time()
        for _ in range(num_runs):
            out = compiled_forward(model, dummy_inputs) if mode == "prefill" else compiled_generate(model, dummy_inputs)
            if isinstance(jax, _JaxBlockUntilReadyProtocol):
                jax.block_until_ready(out)
        end_time = time.time()

        total_time_ms = (end_time - start_time) * 1000.0
        latency_ms = total_time_ms / max(num_runs, 1)

        if mode == "prefill":
            tokens_per_sec = 32 * batch_size * num_runs / max(end_time - start_time, 1e-09)
        else:
            tokens_per_sec = max_new_tokens * batch_size * num_runs / max(end_time - start_time, 1e-09)

        memory_mb = get_current_rss_mb()
        try:
            stats = device.memory_stats() or {}  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
            memory_mb = float(stats.get("peak_bytes_in_use", memory_mb * 1024.0 * 1024.0)) / (1024.0 * 1024.0)
        except (AttributeError, KeyError, RuntimeError, TypeError):
            memory_mb = get_current_rss_mb()

    return (float(tokens_per_sec), float(latency_ms), float(memory_mb))


def benchmark_model(
    model_name: str,
    hardware: str = "tpu-v5p",
    batch_size: int = 32,
    **kwargs: JSONValue,
) -> BenchmarkResultDict:
    """Benchmark a model using the JAX backend.

    Args:
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., `'gpu'`, `'tpu'`, `'cpu'`).
        batch_size: Batch size to use during benchmarking.
        **kwargs: Additional args like `num_runs`, `dtype`, `mode`, `max_new_tokens`, `warmup_steps`.

    Returns:
        A dictionary containing benchmark metrics and status.

    Raises:
        DependencyMissingError: If required JAX dependencies are missing.

    """

    def _run() -> tuple[float, float, float]:
        """Instantiate model and execute JAX benchmark pass.

        Returns:
            Tuple of `(tokens_per_sec, latency_ms, memory_mb)`.

        """
        dtype_str = str(kwargs.get("dtype", "bfloat16"))
        mode = str(kwargs.get("mode", "prefill"))
        max_new_tokens = int(str(kwargs.get("max_new_tokens", 128)))
        warmup_steps = int(str(kwargs.get("warmup_steps", 5)))

        target_dtype = _resolve_jax_dtype(dtype_str)

        config = Gemma4Config.gemma4_e2b()
        config.dtype = target_dtype
        model = Gemma4ForCausalLM(config, rngs=nnx.Rngs(0))

        device = _get_device(hardware)

        num_runs = int(str(kwargs.get("num_runs", 5)))
        return _run_benchmark_pass(model, batch_size, num_runs, warmup_steps, mode, max_new_tokens, device)

    if jax is None or jnp is None or nnx is None or Gemma4ForCausalLM is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise DependencyMissingError("JAX dependencies are missing.")

    return run_benchmark_wrapper(
        backend_name="jax",
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        benchmark_fn=_run,
    )


__all__ = [
    "Gemma4Config",
    "Gemma4ForCausalLM",
    "JaxArrayProtocol",
    "JaxDeviceProtocol",
    "benchmark_model",
]
