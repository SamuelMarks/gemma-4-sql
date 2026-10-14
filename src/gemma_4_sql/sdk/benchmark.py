"""Benchmarking SDK module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.sdk.registry import get_backend

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import BenchmarkResultDict


def benchmark(
    model_name: str,
    hardware: str,
    batch_size: int,
    backend: str,
    dtype: str = "bfloat16",
    mode: str = "prefill",
    max_new_tokens: int = 128,
    warmup_steps: int = 5,
    **kwargs: object,
) -> BenchmarkResultDict:
    """Benchmarks a model on specific hardware.

    Args:
        model_name: The name of the target model.
        hardware: The target hardware accelerator.
        batch_size: The number of items to process in a single batch.
        backend: The backend framework to use.
        dtype: The precision.
        mode: The mode.
        max_new_tokens: Max new tokens.
        warmup_steps: Number of warmup steps.
        **kwargs: Additional parameters including multimodal options.

    Returns:
        A dictionary containing the results.

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
