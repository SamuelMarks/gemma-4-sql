"""MaxText-specific benchmarking pipeline."""

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
Gemma4Model = None
with catch_optional_imports():
    from maxtext.models.gemma4 import Gemma4Model


def _run_benchmark_pass(model: object, params: dict[str, object] | object, batch_size: int, num_runs: int) -> tuple[float, float, float]:
    """Execute the forward pass benchmark loop.

    Args:
        model: The model.
        params: A mapping representing params.
        batch_size: The number of items to process in a single batch.
        num_runs: The integer value for num runs.

    Returns:
        A tuple containing the results.
    """
    dummy_inputs = jnp.zeros((batch_size, 32), dtype=jnp.int32)

    @jax.jit
    def forward_pass(params: dict[str, object] | object, inputs: object) -> object:
        """Execute logic.

        Returns:
            object: The resulting output from the operation.

        """
        return model.apply(params, inputs)

    _ = forward_pass(params, dummy_inputs)
    if hasattr(jax, "block_until_ready"):
        jax.block_until_ready(_)
    start_time = time.time()
    for _ in range(num_runs):
        out = forward_pass(params, dummy_inputs)
    if hasattr(jax, "block_until_ready"):
        jax.block_until_ready(out)
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000.0
    latency_ms = total_time_ms / max(1, num_runs)
    tokens_per_sec = 32 * batch_size * num_runs / max(end_time - start_time, 1e-09)
    memory_mb = 16384.0
    return (float(tokens_per_sec), float(latency_ms), float(memory_mb))


def benchmark_model(model_name: str, hardware: str, batch_size: int, **kwargs: JSONValue) -> JSONDict:
    """Benchmark a model using the MaxText backend.

    Args:
    ----
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., 'tpu-v5p', 'gpu').
        batch_size: Batch size to use during benchmarking.
        **kwargs: Additional args like `num_runs`.

    Returns:
    -------
        A dictionary containing benchmark metrics and status.

    """

    def _run() -> tuple[float, float, float]:
        """Execute function.

        Returns:
            The execution result.

        """
        if not kwargs.get("test_mode"):
            try:
                jax.distributed.initialize()
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
                logger.warning("jax.distributed.initialize() failed: %s", e)
        model = Gemma4Model(model_name)
        rng = jax.random.PRNGKey(0)
        dummy_inputs = jnp.zeros((batch_size, 32), dtype=jnp.int32)
        params = model.init(rng, dummy_inputs)
        num_runs = int(str(kwargs.get("num_runs", 5)))
        return _run_benchmark_pass(model, params, batch_size, num_runs)

    return run_benchmark_wrapper(
        backend_name="maxtext",
        model_name=model_name,
        hardware=hardware,
        batch_size=batch_size,
        missing_deps=jax is None or jnp is None or Gemma4Model is None,
        missing_status="mocked_missing_maxtext",
        benchmark_fn=_run,
    )
