"""Benchmarking SDK module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict


def benchmark(model_name: str, hardware: str, batch_size: int, backend: str) -> JSONDict:
    """Benchmarks a model on specific hardware.

    Args:
        model_name: The name of the target model.
        hardware: The target hardware accelerator.
        batch_size: The number of items to process in a single batch.
        backend: The backend framework to use.

    Returns:
        A dictionary containing the results.
    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).benchmark_model(model_name, hardware, batch_size)
