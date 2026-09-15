"""JAX-specific benchmarking pipeline."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_benchmark import run_benchmark_wrapper

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)

try:
    import jax as _jax
    import jax.numpy as _jnp
    from flax import nnx as _nnx

    from .gemma4 import Gemma4Config as _Gemma4Config
    from .gemma4 import Gemma4ForCausalLM as _Gemma4ForCausalLM

    jax: Any = _jax
    jnp: Any = _jnp
    nnx: Any = _nnx
    Gemma4Config: Any = _Gemma4Config
    Gemma4ForCausalLM: Any = _Gemma4ForCausalLM
except (ImportError, AttributeError):
    jax = None
    jnp = None
    nnx = None
    Gemma4Config = None
    Gemma4ForCausalLM = None


def _get_device(hardware: str) -> Any:
    """Get the jax device for the hardware.

    Args:
        hardware: Target hardware type.

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


def _run_benchmark_pass(model: Any, batch_size: int, num_runs: int, warmup_steps: int, mode: str, max_new_tokens: int, device: Any) -> tuple[float, float, float]:
    """Execute the forward pass benchmark loop.

    Args:
        model: The model.
        batch_size: The number of items to process in a single batch.
        num_runs: The integer value for num runs.
        warmup_steps: Number of warmup steps.
        mode: Benchmark mode.
        max_new_tokens: Max tokens.
        device: JAX device.

    Returns:
        A tuple containing the results.
    """

    def forward_pass(model: Any, inputs: Any) -> Any:
        """Execute a single forward pass.

        Args:
            model: The model to run.
            inputs: Dummy inputs.

        Returns:
            The model forward pass output.
        """
        return model(inputs)

    def generate_pass(model: Any, inputs: Any) -> Any:
        """Execute a simple generation pass.

        Args:
            model: The model to run.
            inputs: Input sequence tokens.

        Returns:
            The generated token sequence.
        """
        seq = inputs
        # A simple unrolled loop for benchmarking generation throughput
        for _ in range(max_new_tokens):
            positions = jnp.arange(seq.shape[1])[None, :]
            logits = model(seq, positions)
            token = jnp.argmax(logits[..., -1, :], axis=-1, keepdims=True)
            seq = jnp.concatenate([seq, token], axis=-1)
        return seq

    if nnx is not None and hasattr(nnx, "jit"):
        forward_pass = nnx.jit(forward_pass)
        generate_pass = nnx.jit(generate_pass)

    with jax.default_device(device):
        dummy_inputs = jax.random.randint(jax.random.key(42), (batch_size, 32), 1, 256000, dtype=jnp.int32)

        # Warmup
        for _ in range(warmup_steps):
            if mode == "prefill":
                out = forward_pass(model, dummy_inputs)
            else:
                out = generate_pass(model, dummy_inputs)
            if hasattr(jax, "block_until_ready"):
                jax.block_until_ready(out)

        start_time = time.time()
        for _ in range(num_runs):
            if mode == "prefill":
                out = forward_pass(model, dummy_inputs)
            else:
                out = generate_pass(model, dummy_inputs)
            # Sync inside loop for proper per-iteration latency
            if hasattr(jax, "block_until_ready"):
                jax.block_until_ready(out)
        end_time = time.time()

        total_time_ms = (end_time - start_time) * 1000.0
        latency_ms = total_time_ms / max(num_runs, 1)

        if mode == "prefill":
            tokens_per_sec = 32 * batch_size * num_runs / max(end_time - start_time, 1e-09)
        else:
            tokens_per_sec = max_new_tokens * batch_size * num_runs / max(end_time - start_time, 1e-09)

        try:
            stats = device.memory_stats()
            memory_mb = stats.get("peak_bytes_in_use", 8192.0 * 1024 * 1024) / (1024 * 1024)
        except (AttributeError, KeyError, RuntimeError, TypeError):
            memory_mb = 8192.0

    return (float(tokens_per_sec), float(latency_ms), float(memory_mb))


def benchmark_model(model_name: str, hardware: str, batch_size: int, **kwargs: JSONValue) -> JSONDict:
    """Benchmark a model using the JAX backend.

    Args:
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., 'gpu', 'tpu', 'cpu').
        batch_size: Batch size to use during benchmarking.
        **kwargs: Additional args like `num_runs`.

    Returns:
        A dictionary containing benchmark metrics and status.

    Raises:
        DependencyMissingError: If required JAX dependencies are missing.
    """

    def _run() -> tuple[float, float, float]:
        """Execute function.

        Returns:
            The execution result.

        """
        dtype_str = str(kwargs.get("dtype", "bfloat16"))
        mode = str(kwargs.get("mode", "prefill"))
        max_new_tokens = int(str(kwargs.get("max_new_tokens", 128)))
        warmup_steps = int(str(kwargs.get("warmup_steps", 5)))

        target_dtype = getattr(jnp, dtype_str, getattr(jnp, "bfloat16", "bfloat16"))

        config = Gemma4Config.gemma4_e2b()
        config.dtype = target_dtype
        model = Gemma4ForCausalLM(config, rngs=nnx.Rngs(0))

        device = _get_device(hardware)

        # JAX doesn't have an easy way to cast all variables of a NNX model without nnx.tree.map
        # But we pass it to config.

        num_runs = int(str(kwargs.get("num_runs", 5)))
        return _run_benchmark_pass(model, batch_size, num_runs, warmup_steps, mode, max_new_tokens, device)

    if jax is None or jnp is None or nnx is None or Gemma4ForCausalLM is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX dependencies are missing.")

    return run_benchmark_wrapper(
        backend_name="jax",
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        missing_deps=False,
        missing_status="missing_jax",
        benchmark_fn=_run,
    )
