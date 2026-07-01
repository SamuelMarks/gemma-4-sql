"""Benchmarking SDK module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict


def benchmark(model_name: str, hardware: str, batch_size: int, backend: str) -> JSONDict:
    """Benchmarks a model on specific hardware.

    Args:
    ----
        model_name: Name of the model.
        hardware: Hardware to benchmark on ('gpu', 'tpu', 'cpu').
        batch_size: Batch size.
        backend: Backend to use.

    Returns:
    -------
        Benchmarking metrics.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).benchmark_model(model_name, hardware, batch_size)
