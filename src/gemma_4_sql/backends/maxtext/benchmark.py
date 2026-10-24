"""MaxText-specific benchmarking pipeline."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_benchmark import get_current_rss_mb, run_benchmark_wrapper
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.type_hints import MaxTextCausalLMProtocol, TensorType

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import BenchmarkResultDict, JSONValue
logger = logging.getLogger(__name__)

jax = optional_import("jax")
jnp = optional_import("jax.numpy")
Gemma4Model = optional_attr("maxtext.models.gemma4", "Gemma4Model")


def _get_device(hardware: str) -> Any:
    """Provide the jax device for the hardware.

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


def _run_benchmark_pass(model: MaxTextCausalLMProtocol, params: dict[str, TensorType], batch_size: int, num_runs: int, warmup_steps: int, mode: str, max_new_tokens: int, device: str | object) -> tuple[float, float, float]:
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
        A tuple containing the results.

    """

    def forward_pass(params: dict[str, TensorType], inputs: TensorType) -> Any:
        """Execute model forward pass.

        Args:
            params: Model parameters.
            inputs: Dummy inputs array.

        Returns:
            Forward pass model output.

        """
        return model.apply(params, inputs)

    def generate_pass(params: dict[str, TensorType], inputs: TensorType) -> Any:
        """Execute simple generation pass.

        Args:
            params: Model parameters.
            inputs: Input sequence array.

        Returns:
            Generated sequence array.

        """
        seq = inputs
        for _ in range(max_new_tokens):
            logits = model.apply(params, seq)
            token = jnp.argmax(logits[..., -1, :], axis=-1, keepdims=True)
            seq = jnp.concatenate([seq, token], axis=-1)
        return seq

    if hasattr(jax, "jit"):
        forward_pass = jax.jit(forward_pass)
        generate_pass = jax.jit(generate_pass)

    with jax.default_device(device):
        dummy_inputs = jax.random.randint(jax.random.PRNGKey(42), (batch_size, 32), 1, 256000, dtype=jnp.int32)

        for _ in range(warmup_steps):
            if mode == "prefill":
                out = forward_pass(params, dummy_inputs)
            else:
                out = generate_pass(params, dummy_inputs)
            if hasattr(jax, "block_until_ready"):
                jax.block_until_ready(out)

        start_time = time.time()
        for _ in range(num_runs):
            if mode == "prefill":
                out = forward_pass(params, dummy_inputs)
            else:
                out = generate_pass(params, dummy_inputs)
            if hasattr(jax, "block_until_ready"):
                jax.block_until_ready(out)
        end_time = time.time()

        total_time_ms = (end_time - start_time) * 1000.0
        latency_ms = total_time_ms / max(1, num_runs)

        if mode == "prefill":
            tokens_per_sec = 32 * batch_size * num_runs / max(end_time - start_time, 1e-09)
        else:
            tokens_per_sec = max_new_tokens * batch_size * num_runs / max(end_time - start_time, 1e-09)

        try:
            dev_any: Any = device
            stats = dev_any.memory_stats()
            memory_mb = stats.get("peak_bytes_in_use", get_current_rss_mb() * 1024 * 1024) / (1024 * 1024)
        except (AttributeError, KeyError, RuntimeError, TypeError):
            memory_mb = get_current_rss_mb()

    return (float(tokens_per_sec), float(latency_ms), float(memory_mb))


def benchmark_model(model_name: str, hardware: str = "tpu-v5p", batch_size: int = 32, **kwargs: JSONValue) -> BenchmarkResultDict:
    """Benchmark a model using the MaxText backend.

    Args:
        model_name: The name of the model to benchmark.
        hardware: Target hardware for the benchmark (e.g., 'tpu-v5p', 'gpu').
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
            model = Gemma4Model(model_name, dtype=getattr(jnp, dtype_str, None))
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
