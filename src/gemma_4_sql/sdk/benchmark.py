"""Benchmarking SDK module."""

from __future__ import annotations

from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.type_hints import BenchmarkResultDict, JSONValue


def benchmark(
    model_name: str,
    hardware: str,
    batch_size: int,
    backend: str,
    dtype: str = "bfloat16",
    mode: str = "prefill",
    max_new_tokens: int = 128,
    warmup_steps: int = 5,
    **kwargs: JSONValue,
) -> BenchmarkResultDict:
    """Benchmark a Text-to-SQL model on a target hardware accelerator.

    Args:
        model_name: Hugging Face model identifier or preset name.
        hardware: Target hardware accelerator (e.g., `'tpu-v5p'`, `'gpu'`, `'cpu'`, `'apple_silicon'`).
        batch_size: Number of sequences processed per benchmark batch.
        backend: Backend framework (`'jax'`, `'pytorch'`, `'keras'`, `'maxtext'`, or `'mlx'`).
        dtype: Numerical precision policy (e.g., `'bfloat16'`, `'float16'`, `'float32'`).
        mode: Benchmark execution mode (`'prefill'` or `'generate'`).
        max_new_tokens: Maximum number of tokens generated in `'generate'` mode.
        warmup_steps: Number of untimed warmup iterations before latency measurement.
        **kwargs: Additional parameters including multimodal options and `num_runs`.

    Returns:
        Structured `BenchmarkResultDict` containing throughput (`tokens_per_sec`), latency, memory, and status.

    """

    return get_backend(backend).benchmark_model(
        model_name,
        hardware,
        batch_size,
        dtype=dtype,
        mode=mode,
        max_new_tokens=max_new_tokens,
        warmup_steps=warmup_steps,
        backend_alias=backend,
        **kwargs,
    )


__all__ = ["benchmark"]
