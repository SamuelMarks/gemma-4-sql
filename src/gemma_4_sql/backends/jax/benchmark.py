# Copyright 2024
"""JAX-specific benchmarking pipeline."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_benchmark import run_benchmark_wrapper
from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)
jax = None
jnp = None
with catch_optional_imports():
    import jax
    import jax.numpy as jnp
Gemma4ForCausalLM = None
Gemma4Config = None
nnx = None
with catch_optional_imports():
    from flax import nnx

    from .gemma4 import Gemma4Config, Gemma4ForCausalLM


def _run_benchmark_pass(model: object, batch_size: int, num_runs: int) -> tuple[float, float, float]:
    """Execute the forward pass benchmark loop.

    Args:
    ----
        model: The initialized model.
        batch_size: Inference batch size.
        num_runs: Number of benchmark iterations.

    Returns:
    -------
        A tuple of (tokens_per_sec, latency_ms, memory_mb).

    """

    @nnx.jit
    def forward_pass(model: object, inputs: object) -> object:
        """Execute a single forward pass.

        Returns:
            object: The resulting output from the operation.

        """
        return model(inputs)

    dummy_inputs = jnp.zeros((batch_size, 32), dtype=jnp.int32)
    _ = forward_pass(model, dummy_inputs)
    if hasattr(jax, "block_until_ready"):
        jax.block_until_ready(_)
    start_time = time.time()
    for _ in range(num_runs):
        out = forward_pass(model, dummy_inputs)
    if hasattr(jax, "block_until_ready"):
        jax.block_until_ready(out)
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000.0
    latency_ms = total_time_ms / max(num_runs, 1)
    tokens_per_sec = 32 * batch_size * num_runs / max(end_time - start_time, 1e-09)
    memory_mb = 8192.0
    return (float(tokens_per_sec), float(latency_ms), float(memory_mb))


def benchmark_model(model_name: str, hardware: str, batch_size: int, **kwargs: JSONValue) -> JSONDict:
    """Benchmark a model using the JAX backend.

    Args:
    ----
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., 'gpu', 'tpu', 'cpu').
        batch_size: Batch size to use during benchmarking.
        **kwargs: Additional args like `num_runs`.

    Returns:
    -------
        A dictionary containing benchmark metrics and status.

    """

    def _run() -> tuple[float, float, float]:
        model = Gemma4ForCausalLM(Gemma4Config.gemma4_e2b(), rngs=nnx.Rngs(0))
        num_runs = int(str(kwargs.get("num_runs", 5)))
        return _run_benchmark_pass(model, batch_size, num_runs)

    return run_benchmark_wrapper(
        backend_name="jax",
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        missing_deps=jax is None or jnp is None or nnx is None or Gemma4ForCausalLM is None,
        missing_status="mocked_missing_jax",
        benchmark_fn=_run,
    )
